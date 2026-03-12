try:
    import PyInstaller
    import PyInstaller.__main__ as m
    print('OK', PyInstaller.__version__)
except Exception as e:
    print('ERR', repr(e))
