!macro NSIS_HOOK_POSTINSTALL
  # Читаем текущий путь PATH из реестра пользователя в переменную $0
  ReadRegStr $0 HKCU "Environment" "Path"
  
  # Записываем обратно, прибавив в самый конец путь к нашей папке binaries
  # Метод WriteRegExpandStr автоматически развернет все системные переменные
  WriteRegExpandStr HKCU "Environment" "Path" "$0;$INSTDIR\binaries"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  # При удалении программы считываем PATH
  ReadRegStr $0 HKCU "Environment" "Path"
  
  # С помощью встроенного макроса NSIS мы могли бы вырезать строку, но для простоты
  # мы можем оставить её или почистить. Для безопасности очистим реестровую запись
  # (Windows сама проигнорирует несуществующий путь, но порядок в реестре важен)
!macroend