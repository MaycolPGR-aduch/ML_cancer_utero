@echo off
cd /d "%~dp0"
echo ====================================================================
echo  Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Servidor Web FastAPI
echo  Abre en tu navegador: http://127.0.0.1:8000
echo ====================================================================
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
