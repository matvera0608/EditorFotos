@echo off

git rm -rf --cached .

git reset --soft origin/main
echo Restaurando todo a la normalidad
git add .
git commit -m "Versión limpia sin venvs"
pause
git push -f origin main
pause