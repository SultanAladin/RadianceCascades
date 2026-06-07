[CmdletBinding()]
param(
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------------
# Patterned on BuildSolidArc.ps1 (C:\Users\OS\Documents\TheGreatFoundation\
# SolidArc - Copy (2)\BuildSolidArc.ps1). Differences:
#   * No LunaSVG/PlutoVG.
#   * Two output targets: RenderingSubsystem.lib (Source/* + ImGui) AND
#     RenderingSubsystem.exe (Application/* linked against the .lib).
#   * ExternalPackages is resolved against both ..\ and ..\..\  because this
#     project is nested one level deeper (RayArc\) than SolidArc.
# ----------------------------------------------------------------------------

function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" }
function Write-Warn { param([string]$Message) Write-Host "[WARN] $Message" }
function Write-Err  { param([string]$Message) Write-Host "[ERROR] $Message" }

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Resolve-FirstExistingPath {
    param([string]$BaseDirectory, [string[]]$Candidates, [string]$Label)
    foreach ($candidate in $Candidates) {
        $candidatePath = $candidate
        if (-not [System.IO.Path]::IsPathRooted($candidatePath)) {
            $candidatePath = Join-Path $BaseDirectory $candidatePath
        }
        $candidatePath = [System.IO.Path]::GetFullPath($candidatePath)
        if (Test-Path -LiteralPath $candidatePath) { return $candidatePath }
    }
    throw "$Label not found."
}

function Import-VcVarsEnvironment {
    param([string]$VcVarsPath)
    Write-Info "Using MSVC: $VcVarsPath"
    # VS Insiders' vcvars64.bat shells out to vswhere.exe but doesn't extend
    # PATH first; prepend the Installer directory so the call succeeds.
    $vsWhereDirs = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\Installer",
        "C:\Program Files\Microsoft Visual Studio\Installer"
    ) | Where-Object { Test-Path -LiteralPath $_ }
    $pathPrefix = ""
    foreach ($d in $vsWhereDirs) { $pathPrefix += "$d;" }
    $command = "set `"PATH=$pathPrefix%PATH%`" && call `"$VcVarsPath`" >nul && set"
    $envDump = & cmd.exe /d /s /c $command
    if ($LASTEXITCODE -ne 0) { throw "Failed to import Visual Studio environment from: $VcVarsPath" }
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    if (-not (Get-Command cl.exe   -ErrorAction SilentlyContinue)) { throw "cl.exe was not found after loading vcvars."   }
    if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) { throw "link.exe was not found after loading vcvars." }
    if (-not (Get-Command lib.exe  -ErrorAction SilentlyContinue)) { throw "lib.exe was not found after loading vcvars."  }
}

function Get-Sha256 {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hash = $sha.ComputeHash($bytes)
        return ($hash | ForEach-Object { $_.ToString("x2") }) -join ""
    } finally { $sha.Dispose() }
}

function Test-PathUnderAnyRoot {
    param([string]$Path, [string[]]$Roots)
    foreach ($root in $Roots) {
        if ($Path.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-SourceOutputKey {
    param([string]$SourcePath, [string]$ProjectRoot, [string]$ImguiRoot)
    $fullSourcePath = [System.IO.Path]::GetFullPath($SourcePath)
    if ($fullSourcePath.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullSourcePath.Substring($ProjectRoot.Length).TrimStart("\", "/")
    }
    if ($fullSourcePath.StartsWith($ImguiRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $fullSourcePath.Substring($ImguiRoot.Length).TrimStart("\", "/")
        return Join-Path "External\imgui" $relative
    }
    $sanitized = $fullSourcePath -replace "[:\\/\s]", "_"
    return Join-Path "External\misc" $sanitized
}

function Get-CompileReason {
    param([string]$ObjectPath, [string]$DependencyFile, [string]$CommandFile, [string]$CompileSignature)
    if (-not (Test-Path -LiteralPath $ObjectPath))     { return "missing object"             }
    if (-not (Test-Path -LiteralPath $DependencyFile)) { return "missing dependency cache"   }
    if (-not (Test-Path -LiteralPath $CommandFile))    { return "missing command signature"  }
    $storedSignature = (Get-Content -LiteralPath $CommandFile -Raw).Trim()
    if ($storedSignature -ne $CompileSignature) { return "compile settings changed" }
    $objectTime = (Get-Item -LiteralPath $ObjectPath).LastWriteTimeUtc
    $dependencyPaths = @(Get-Content -LiteralPath $DependencyFile | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
    if ($dependencyPaths.Count -eq 0) { return "empty dependency cache" }
    foreach ($dependencyPath in $dependencyPaths) {
        if (-not (Test-Path -LiteralPath $dependencyPath)) { return "dependency removed: $dependencyPath" }
        $dependencyTime = (Get-Item -LiteralPath $dependencyPath).LastWriteTimeUtc
        if ($dependencyTime -gt $objectTime) { return "dependency updated: $dependencyPath" }
    }
    return $null
}

function Get-LinkReason {
    param([string]$ExecutablePath, [string[]]$ObjectPaths, [string]$LinkSignatureFile, [string]$LinkSignature)
    if (-not (Test-Path -LiteralPath $ExecutablePath))    { return "missing output"        }
    if (-not (Test-Path -LiteralPath $LinkSignatureFile)) { return "missing link signature"}
    $storedSignature = (Get-Content -LiteralPath $LinkSignatureFile -Raw).Trim()
    if ($storedSignature -ne $LinkSignature) { return "link settings changed" }
    $exeTime = (Get-Item -LiteralPath $ExecutablePath).LastWriteTimeUtc
    foreach ($objectPath in $ObjectPaths) {
        if (-not (Test-Path -LiteralPath $objectPath)) { return "missing object: $objectPath" }
        $objectTime = (Get-Item -LiteralPath $objectPath).LastWriteTimeUtc
        if ($objectTime -gt $exeTime) { return "updated object: $objectPath" }
    }
    return $null
}

function Get-ProcessLockingPath {
    param([string]$Path)
    $normalizedPath = [System.IO.Path]::GetFullPath($Path)
    foreach ($process in Get-Process -ErrorAction SilentlyContinue) {
        try {
            if ($process.Path -and ([System.IO.Path]::GetFullPath($process.Path).Equals($normalizedPath, [System.StringComparison]::OrdinalIgnoreCase))) {
                return $process
            }
        } catch { continue }
    }
    return $null
}

function Find-ShaderCompiler {
    param([string]$VulkanDirectory)
    if ($env:VULKAN_SDK) {
        $fromEnv = Join-Path $env:VULKAN_SDK "Bin\glslangValidator.exe"
        if (Test-Path -LiteralPath $fromEnv) { return $fromEnv }
    }
    $localPath = Join-Path $VulkanDirectory "Bin\glslangValidator.exe"
    if (Test-Path -LiteralPath $localPath) { return $localPath }
    if (Test-Path -LiteralPath "C:\VulkanSDK") {
        $candidate = Get-ChildItem -Path "C:\VulkanSDK" -Directory |
            Sort-Object -Property Name -Descending |
            ForEach-Object { Join-Path $_.FullName "Bin\glslangValidator.exe" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
        if ($candidate) { return $candidate }
    }
    $fromPath = Get-Command glslangValidator.exe -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    return $null
}

function Compile-ShaderIfNeeded {
    param(
        [string]$ShaderCompiler,
        [string]$SourcePath,
        [string]$OutputPath,
        [string]$IncludeDir = "",
        [datetime]$LatestIncludeTimeUtc
    )
    $needsCompile = $false
    if (-not (Test-Path -LiteralPath $OutputPath)) {
        $needsCompile = $true
    } else {
        $sourceTime = (Get-Item -LiteralPath $SourcePath).LastWriteTimeUtc
        $outputTime = (Get-Item -LiteralPath $OutputPath).LastWriteTimeUtc
        if ($sourceTime -gt $outputTime) { $needsCompile = $true }
        elseif ($LatestIncludeTimeUtc -and $LatestIncludeTimeUtc -gt $outputTime) {
            # Any include changed → recompile. Cheap insurance: glslangValidator
            # has no built-in dep tracking, so we mtime-fan the include dir.
            $needsCompile = $true
        }
    }
    if (-not $needsCompile) {
        Write-Host "[SKIP]  Shader $(Split-Path $SourcePath -Leaf)"
        return
    }
    Write-Host "[BUILD] Shader $(Split-Path $SourcePath -Leaf)"
    $compileArgs = @("-V", $SourcePath, "-o", $OutputPath)
    if ($IncludeDir -and (Test-Path -LiteralPath $IncludeDir)) {
        $compileArgs += @("-I$IncludeDir")
    }
    $shaderOutput = & $ShaderCompiler @compileArgs 2>&1
    foreach ($line in $shaderOutput) {
        if ($line) { Write-Host "       $line" }
    }
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $SourcePath" }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

try {
    # Locate ExternalPackages. SolidArc lives at TheGreatFoundation\<project>\
    # and references ..\ExternalPackages; RenderingSubsystem lives at
    # TheGreatFoundation\RayArc\<project>\ so we try ..\.. first, then ..\.
    $vulkanDir = Resolve-FirstExistingPath -BaseDirectory $scriptRoot -Candidates @(
        "..\..\ExternalPackages\vulkan",
        "..\ExternalPackages\vulkan"
    ) -Label "Vulkan SDK"
    $glmDir = Resolve-FirstExistingPath -BaseDirectory $scriptRoot -Candidates @(
        "..\..\ExternalPackages\glm",
        "..\ExternalPackages\glm"
    ) -Label "GLM"
    $imguiDir = Resolve-FirstExistingPath -BaseDirectory $scriptRoot -Candidates @(
        "..\..\ExternalPackages\imgui",
        "..\ExternalPackages\imgui"
    ) -Label "ImGui"

    Write-Info "Vulkan SDK: $vulkanDir"
    Write-Info "GLM: $glmDir"
    Write-Info "ImGui: $imguiDir"

    $buildDir         = $scriptRoot
    $shaderOutDir     = Join-Path $buildDir "Artifacts\Shaders"
    $logsDir          = Join-Path $buildDir "Logs"
    $objDir           = Join-Path $buildDir "Artifacts\obj-adaptive"
    $stateDir         = Join-Path $objDir ".state"
    $libPath          = Join-Path $buildDir "RenderingSubsystem.lib"
    $exePath          = Join-Path $buildDir "RenderingSubsystem.exe"
    $libSignatureFile = Join-Path $stateDir "lib.cmd"
    $linkSignatureFile= Join-Path $stateDir "link.cmd"

    if ($Clean) {
        Write-Info "Cleaning adaptive artifacts..."
        if (Test-Path -LiteralPath $objDir) { Remove-Item -LiteralPath $objDir -Recurse -Force }
        if (Test-Path -LiteralPath $libPath) { Remove-Item -LiteralPath $libPath -Force }
        if (Test-Path -LiteralPath $exePath) { Remove-Item -LiteralPath $exePath -Force }
    }

    Ensure-Directory $buildDir
    Ensure-Directory $shaderOutDir
    Ensure-Directory $logsDir
    Ensure-Directory $objDir
    Ensure-Directory $stateDir

    # NOTE: VS 18 Insiders' vcvars64.bat assumes vswhere.exe on PATH and breaks
    # in non-interactive shells; prefer VS 2022 Community/Pro/Ent first.
    $vcvarsPath = Resolve-FirstExistingPath -BaseDirectory $scriptRoot -Candidates @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) -Label "Visual Studio vcvars64.bat"
    Import-VcVarsEnvironment -VcVarsPath $vcvarsPath

    # -------------------- shaders --------------------
    $shaderCompiler = Find-ShaderCompiler -VulkanDirectory $vulkanDir
    if ($shaderCompiler) {
        Write-Info "Using shader compiler: $shaderCompiler"
        $shaderDir   = Join-Path $scriptRoot "Shaders"
        $includeDir  = Join-Path $shaderDir "include"
        # Newest mtime in the include tree. glslangValidator has no dep file, so
        # we treat "any include changed" as "every .comp/.vert/.frag may have
        # changed" and let the per-file mtime check below short-circuit.
        $latestInc = [datetime]::MinValue
        if (Test-Path -LiteralPath $includeDir) {
            $incFiles = Get-ChildItem -LiteralPath $includeDir -Recurse -File `
                -Include *.glsl,*.h,*.glslh -ErrorAction SilentlyContinue
            foreach ($f in $incFiles) {
                if ($f.LastWriteTimeUtc -gt $latestInc) { $latestInc = $f.LastWriteTimeUtc }
            }
        }
        if (Test-Path -LiteralPath $shaderDir) {
            Get-ChildItem -LiteralPath $shaderDir -Filter "*.comp" -File | ForEach-Object {
                $output = Join-Path $shaderOutDir ("{0}.spv" -f $_.BaseName)
                Compile-ShaderIfNeeded -ShaderCompiler $shaderCompiler -SourcePath $_.FullName `
                    -OutputPath $output -IncludeDir $shaderDir -LatestIncludeTimeUtc $latestInc
            }
            Get-ChildItem -LiteralPath $shaderDir -Filter "*.vert" -File | ForEach-Object {
                $output = Join-Path $shaderOutDir ("{0}_vert.spv" -f $_.BaseName)
                Compile-ShaderIfNeeded -ShaderCompiler $shaderCompiler -SourcePath $_.FullName `
                    -OutputPath $output -IncludeDir $shaderDir -LatestIncludeTimeUtc $latestInc
            }
            Get-ChildItem -LiteralPath $shaderDir -Filter "*.frag" -File | ForEach-Object {
                $output = Join-Path $shaderOutDir ("{0}_frag.spv" -f $_.BaseName)
                Compile-ShaderIfNeeded -ShaderCompiler $shaderCompiler -SourcePath $_.FullName `
                    -OutputPath $output -IncludeDir $shaderDir -LatestIncludeTimeUtc $latestInc
            }
        }
    } else {
        Write-Warn "glslangValidator.exe not found. Skipping shader compilation."
    }

    # -------------------- source lists --------------------
    # Lib sources: every Source\* TU plus the Scene facade.
    $libProjectSourceRelPaths = @(
        "Source\Core\VulkanContext.cpp",
        "Source\Core\FrameGraph.cpp",
        "Source\Core\Logger.cpp",
        "Source\Renderer\Renderer.cpp",
        "Source\Renderer\GBuffer.cpp",
        "Source\Renderer\Lighting.cpp",
        "Source\Renderer\SkyAtmosphere.cpp",
        "Source\Renderer\Tonemap.cpp",
        "Source\Renderer\PerfTimers.cpp",
        "Source\Renderer\OffscreenTargets.cpp",
        "Source\Renderer\GridPass.cpp",
        "Source\Renderer\GBufferPreview.cpp",
        "Source\Renderer\Picking.cpp",
        "Source\Renderer\InstanceXformBuffer.cpp",
        "Source\Shadow\SDFConeShadow.cpp",
        "Source\GI\RadianceCascades.cpp",
        "Source\SDF\MeshSDFBaker.cpp",
        "Source\SDF\GlobalSDF.cpp",
        "Source\SDF\SDFTrace.cpp",
        "Source\SDF\SDFCache.cpp",
        "Source\Material\PbrMaterial.cpp",
        "Source\Material\MaterialSeed.cpp",
        "Source\Material\TextureUpload.cpp",
        "Source\Scene\MeshRegistry.cpp",
        "Source\Scene\InstanceRegistry.cpp",
        "Source\Scene\ObjLoader.cpp",
        "Source\Scene\Scene.cpp",
        "Source\Scene\FloorMesh.cpp",
        "Source\Scene\ScaleRefSphere.cpp",
        "Source\Camera\OrbitCamera.cpp",
        "Source\UI\ImGuiPanel.cpp",
        "Source\UI\ImGuiContextRS.cpp",
        "Source\UI\PerfWidget.cpp",
        "Source\UI\RenderSettingsPanel.cpp",
        "Source\UI\SdfBakerPanel.cpp"
    )
    $imguiSourceRelPaths = @(
        "imgui.cpp",
        "imgui_draw.cpp",
        "imgui_tables.cpp",
        "imgui_widgets.cpp",
        "backends\imgui_impl_win32.cpp",
        "backends\imgui_impl_vulkan.cpp"
    )
    # Exe sources: Application\* only (linked against the .lib).
    $exeSourceRelPaths = @(
        "Application\Main.cpp",
        "Application\Win32Host.cpp"
    )

    $libSourcePaths = @()
    foreach ($relativePath in $libProjectSourceRelPaths) {
        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $relativePath))
        if (-not (Test-Path -LiteralPath $fullPath)) { throw "Missing source file: $fullPath" }
        $libSourcePaths += $fullPath
    }
    foreach ($relativePath in $imguiSourceRelPaths) {
        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $imguiDir $relativePath))
        if (-not (Test-Path -LiteralPath $fullPath)) { throw "Missing source file: $fullPath" }
        $libSourcePaths += $fullPath
    }
    $exeSourcePaths = @()
    foreach ($relativePath in $exeSourceRelPaths) {
        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $relativePath))
        if (-not (Test-Path -LiteralPath $fullPath)) { throw "Missing source file: $fullPath" }
        $exeSourcePaths += $fullPath
    }

    $includeDirs = @(
        (Join-Path $vulkanDir "include"),
        $glmDir,
        $imguiDir,
        (Join-Path $imguiDir "backends"),
        (Join-Path $scriptRoot "Include"),
        (Join-Path $scriptRoot "Source"),
        $scriptRoot
    )
    $includeArgs = $includeDirs | ForEach-Object { "/I$_" }

    $compileArgsCommon = @(
        "/nologo",
        "/EHsc",
        "/std:c++17",
        "/W3",
        "/O2",
        "/DNDEBUG",
        "/DWIN32",
        "/D_WINDOWS",
        # NOTE: ExternalPackages\imgui\imconfig.h already enables math operators.
        # Defining it on the command line as well triggers C4005 redefinition.
        "/showIncludes"
    )

    $libraries = @(
        "user32.lib",
        "gdi32.lib",
        "shell32.lib",
        "comdlg32.lib",
        (Join-Path $vulkanDir "lib\vulkan-1.lib")
    )

    $trackedDependencyRoots = @(
        ([System.IO.Path]::GetFullPath($scriptRoot).TrimEnd("\") + "\"),
        ([System.IO.Path]::GetFullPath($imguiDir).TrimEnd("\") + "\"),
        ([System.IO.Path]::GetFullPath($glmDir).TrimEnd("\") + "\"),
        ([System.IO.Path]::GetFullPath((Join-Path $vulkanDir "include")).TrimEnd("\") + "\")
    )

    $compileSignature = Get-Sha256 -Text (@(
            ($compileArgsCommon -join " "),
            ($includeDirs -join ";"),
            "profile=release"
        ) -join "`n")

    # -------------------- compile helper --------------------
    function Invoke-CompileBatch {
        param([string[]]$SourcePaths)

        $compiled = 0
        $skipped  = 0
        $objects  = New-Object System.Collections.Generic.List[string]
        foreach ($sourcePath in $SourcePaths) {
            $outputKey = Get-SourceOutputKey -SourcePath $sourcePath `
                -ProjectRoot ([System.IO.Path]::GetFullPath($scriptRoot).TrimEnd("\") + "\") `
                -ImguiRoot   ([System.IO.Path]::GetFullPath($imguiDir).TrimEnd("\") + "\")
            $objectPath     = Join-Path $objDir   ([System.IO.Path]::ChangeExtension($outputKey, ".obj"))
            $dependencyFile = Join-Path $stateDir ([System.IO.Path]::ChangeExtension($outputKey, ".deps"))
            $commandFile    = Join-Path $stateDir ([System.IO.Path]::ChangeExtension($outputKey, ".cmd"))

            Ensure-Directory (Split-Path -Parent $objectPath)
            Ensure-Directory (Split-Path -Parent $dependencyFile)
            Ensure-Directory (Split-Path -Parent $commandFile)

            $reason = Get-CompileReason -ObjectPath $objectPath -DependencyFile $dependencyFile `
                -CommandFile $commandFile -CompileSignature $compileSignature
            if ($reason) {
                Write-Host "[BUILD] $outputKey ($reason)"
                $singleCompileArgs = @("/c") + $compileArgsCommon + $includeArgs + @("/Fo$objectPath", $sourcePath)
                $compilerOutput = & cl.exe @singleCompileArgs 2>&1

                $dependencies = New-Object "System.Collections.Generic.HashSet[string]" ([System.StringComparer]::OrdinalIgnoreCase)
                [void]$dependencies.Add([System.IO.Path]::GetFullPath($sourcePath))

                foreach ($line in $compilerOutput) {
                    $text = [string]$line
                    if ($text -match "^\s*Note:\s+including file:\s*(.+)$") {
                        $candidateDependency = $matches[1].Trim()
                        if ($candidateDependency.Length -eq 0) { continue }
                        $fullDependencyPath = $candidateDependency
                        if (-not [System.IO.Path]::IsPathRooted($fullDependencyPath)) {
                            $fullDependencyPath = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $fullDependencyPath))
                        } else {
                            $fullDependencyPath = [System.IO.Path]::GetFullPath($fullDependencyPath)
                        }
                        if ((Test-Path -LiteralPath $fullDependencyPath) -and (Test-PathUnderAnyRoot -Path $fullDependencyPath -Roots $trackedDependencyRoots)) {
                            [void]$dependencies.Add($fullDependencyPath)
                        }
                        continue
                    }
                    if ($text.Trim().Length -gt 0) { Write-Host "       $text" }
                }

                if ($LASTEXITCODE -ne 0) { throw "Compilation failed for: $sourcePath" }

                $dependencies | Sort-Object | Set-Content -LiteralPath $dependencyFile -Encoding UTF8
                Set-Content -LiteralPath $commandFile -Value $compileSignature -Encoding UTF8
                $compiled++
            } else {
                Write-Host "[SKIP]  $outputKey"
                $skipped++
            }
            $objects.Add($objectPath)
        }
        return [pscustomobject]@{
            Objects  = $objects
            Compiled = $compiled
            Skipped  = $skipped
        }
    }

    # -------------------- compile + archive lib --------------------
    Write-Info "Adaptive C++ build started (lib stage)..."
    $libBatch = Invoke-CompileBatch -SourcePaths $libSourcePaths

    $libSignature = Get-Sha256 -Text (@(
            ($libBatch.Objects -join ";"),
            "stage=lib"
        ) -join "`n")
    $libReason = Get-LinkReason -ExecutablePath $libPath -ObjectPaths $libBatch.Objects.ToArray() `
        -LinkSignatureFile $libSignatureFile -LinkSignature $libSignature

    if ($libReason) {
        Write-Info "Archiving RenderingSubsystem.lib ($libReason)"
        $libArgs = @("/nologo", "/OUT:$libPath") + $libBatch.Objects.ToArray()
        $libOutput = & lib.exe @libArgs 2>&1
        foreach ($line in $libOutput) {
            $text = [string]$line
            if ($text.Trim().Length -gt 0) { Write-Host "       $text" }
        }
        if ($LASTEXITCODE -ne 0) { throw "lib.exe failed." }
        Set-Content -LiteralPath $libSignatureFile -Value $libSignature -Encoding UTF8
    } else {
        Write-Host "[SKIP]  Lib archive step (up to date)"
    }

    # -------------------- compile + link exe --------------------
    Write-Info "Adaptive C++ build started (exe stage)..."
    $exeBatch = Invoke-CompileBatch -SourcePaths $exeSourcePaths

    $linkInputs = @($libPath) + $libraries
    $linkSignature = Get-Sha256 -Text (@(
            ($exeBatch.Objects -join ";"),
            ($linkInputs       -join ";"),
            "subsystem=windows"
        ) -join "`n")
    $linkReason = Get-LinkReason -ExecutablePath $exePath -ObjectPaths $exeBatch.Objects.ToArray() `
        -LinkSignatureFile $linkSignatureFile -LinkSignature $linkSignature

    if ($linkReason) {
        $lockingProcess = Get-ProcessLockingPath -Path $exePath
        if ($lockingProcess) {
            throw "Cannot link while RenderingSubsystem.exe is running (PID $($lockingProcess.Id))."
        }
        Write-Info "Linking RenderingSubsystem.exe ($linkReason)"
        $linkArgs = @("/nologo", "/SUBSYSTEM:WINDOWS", "/ENTRY:WinMainCRTStartup", "/OUT:$exePath") + $exeBatch.Objects.ToArray() + @($libPath) + $libraries
        $linkOutput = & link.exe @linkArgs 2>&1
        foreach ($line in $linkOutput) {
            $text = [string]$line
            if ($text.Trim().Length -gt 0) { Write-Host "       $text" }
        }
        if ($LASTEXITCODE -ne 0) { throw "Link failed." }
        Set-Content -LiteralPath $linkSignatureFile -Value $linkSignature -Encoding UTF8
    } else {
        Write-Host "[SKIP]  Link step (up to date)"
    }

    $vulkanDll = Join-Path $vulkanDir "bin\vulkan-1.dll"
    if (Test-Path -LiteralPath $vulkanDll) {
        Copy-Item -LiteralPath $vulkanDll -Destination $buildDir -Force
    }

    Write-Host ""
    Write-Host "[SUCCESS] Build complete: $exePath"
    Write-Host "[INFO] Lib compiled: $($libBatch.Compiled), skipped: $($libBatch.Skipped)"
    Write-Host "[INFO] Exe compiled: $($exeBatch.Compiled), skipped: $($exeBatch.Skipped)"
    exit 0
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}
