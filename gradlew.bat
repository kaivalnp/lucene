@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem
@rem SPDX-License-Identifier: Apache-2.0
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  Gradle startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
@rem This is normally unused
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:execute
@rem Setup the command line

set CLASSPATH=

@rem START OF LUCENE CUSTOMIZATION

@rem LUCENE-9471: workaround for gradle leaving junk temp. files behind.
SET GRADLE_TEMPDIR=%DIRNAME%\.gradle\tmp
IF NOT EXIST "%GRADLE_TEMPDIR%" MKDIR "%GRADLE_TEMPDIR%"
SET DEFAULT_JVM_OPTS=%DEFAULT_JVM_OPTS% "-Djava.io.tmpdir=%GRADLE_TEMPDIR%"

@rem LUCENE-9266: verify and download the gradle wrapper jar if we don't have one.
set GRADLE_WRAPPER_JAR=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar
set GRADLE_WRAPPER_CHECKSUM=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar.sha256

@rem Read the expected hash from .sha256 file
for /f "tokens=1 usebackq" %%A in ("%GRADLE_WRAPPER_CHECKSUM%") do (
    set "EXPECTED=%%A"
)
@rem Get actual SHA-256 hash using certutil
for /f "tokens=* delims=" %%H in ('certutil -hashfile "%GRADLE_WRAPPER_JAR%" SHA256 ^| findstr /R /B /I /X "[0-9a-f]*"') do (
    set "ACTUAL=%%H"
)

if /i "%ACTUAL%" NEQ "%EXPECTED%" (
  "%JAVA_EXE%" -XX:TieredStopAtLevel=1 %JAVA_OPTS% "%APP_HOME%/build-tools/build-infra/src/main/java/org/apache/lucene/gradle/WrapperDownloader.java" "%GRADLE_WRAPPER_JAR%"
  IF %ERRORLEVEL% EQU 1 goto failWithJvmMessage
  IF %ERRORLEVEL% NEQ 0 goto fail
)

@rem Generate gradle.properties if they don't exist
IF NOT EXIST "%APP_HOME%\gradle.properties" (
  @rem local expansion is needed to check ERRORLEVEL inside control blocks.
  setlocal enableDelayedExpansion
  "%JAVA_EXE%" %JAVA_OPTS% "%APP_HOME%/build-tools/build-infra/src/main/java/org/apache/lucene/gradle/GradlePropertiesGenerator.java" "%APP_HOME%\gradle\template.gradle.properties" "%APP_HOME%\gradle.properties"
  IF %ERRORLEVEL% NEQ 0 goto fail
  endlocal
)

@rem END OF LUCENE CUSTOMIZATION

@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=%APP_BASE_NAME%" -classpath "%CLASSPATH%" -jar "%APP_HOME%\gradle\wrapper\gradle-wrapper.jar" %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd
goto fail

:failWithJvmMessage
@rem https://github.com/apache/lucene/pull/819
echo Error: Something went wrong. Make sure you're using Java version of exactly 24.

:fail
rem Set variable GRADLE_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GRADLE_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
