Set WshShell = CreateObject("WScript.Shell")

' Step 1: Kill any existing server
WshShell.Run """D:\PriceSystem_Web\_kill_server.bat""", 0, True

' Step 2: Wait for cleanup
WScript.Sleep 1500

' Step 3: Start server silently
WshShell.Run "cmd /c cd /d D:\PriceSystem_Web & C:\Python313\python.exe server.py", 0, False

' Step 4: Wait for server to start
WScript.Sleep 5000

' Step 5: Open Edge browser
WshShell.Run """C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"" http://127.0.0.1:5000", 1, False
