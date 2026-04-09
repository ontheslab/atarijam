@echo off
set Z88DK_DIR=C:\z88dk
set ZCCCFG=C:\z88dk\lib\config
set PATH=C:\z88dk\bin;%PATH%

zcc +cpm -vn -create-app -compiler=sdcc --opt-code-speed nabujam.c -o NABUJAM
if errorlevel 1 goto fail

copy /y NABUJAM.COM "D:\NIA\NABU Internet Adapter\Store\D\0\"
if errorlevel 1 goto fail

echo.
echo Build OK
dir NABUJAM.COM
exit /b 0

:fail
echo.
echo Build FAILED
exit /b 1
