Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """" & "D:\PriceSystem_Web\_kill_server.bat" & """", 0, True
MsgBox "服务器已停止", vbInformation, "专线估价系统"
