@echo on

set "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPY=%PKG_VERSION%"
set "SETUPTOOLS_SCM_PRETEND_VERSION=%PKG_VERSION%"

%PYTHON% -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit 1

%PYTHON% "%RECIPE_DIR%\strip_pyd_manifests.py" "%PREFIX%\Lib\site-packages"
if errorlevel 1 exit 1
