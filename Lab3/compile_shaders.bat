@echo off
setlocal
cd /d "%~dp0"

if not defined VULKAN_SDK (
    echo ERROR: Variable VULKAN_SDK is not set. Install Vulkan SDK or set VULKAN_SDK manually.
    pause
    exit /b 1
)

set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
if not exist "%GLSLC%" (
    echo ERROR: glslc.exe was not found at "%GLSLC%".
    echo Check your Vulkan SDK installation.
    pause
    exit /b 1
)

for %%S in (shader.vert shader.frag unlit.vert unlit.frag) do (
    echo Compiling shaders\%%S ...
    "%GLSLC%" "shaders\%%S" -o "shaders\%%S.spv"
    if errorlevel 1 (
        echo ERROR: Failed to compile shaders\%%S
        pause
        exit /b 1
    )
)

echo.
echo All shaders compiled successfully.
echo You can now start the application.
pause
