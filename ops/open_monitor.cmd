@echo off
cd /d "%~dp0.."
python -c "from ops.supervisor import open_monitor; open_monitor()"
