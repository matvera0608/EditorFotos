@echo off

git rm -r --cached editor_cuda_env 2>nul
git rm -r --cached editor_dml_env 2>nul

git add .
git commit -m "Version limpia sin venvs, se puede subir normalmente los archivos"
pause
git push -f origin main
pause