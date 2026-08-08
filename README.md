# WIG20-GARCH-Repo

Repozytorium udokumentowujące kod wykorzystany do pracy magisterskiej "Empiryczne własności modeli zmienności w przypadku wysokowymiarowych danych giełdowych". **Skrypt i kod zakładają, że są wykonywane na Windows 11 64-bit (Intel lub AMD, a nie ARM64)**. Środowiska zostały tak zbudowane, aby oferowały największą możliwą powtarzalność bez zakładania globalnych instalacji Python oraz R. Wymagane są jednak [uv](https://github.com/astral-sh/uv/releases/tag/0.11.7) oraz [rig](https://github.com/r-lib/rig/releases/tag/v0.8.1) dla których warto się trzymać wskazanych wersji. Istotne jest, że mimo, że ten plik jest napisany po polsku, to sam kod i komentarze wykorzystują język angielski.

## Środowisko

Aby przygotować środowisko dla Python i R należy wywołać skrypt PowerShell `setup.ps1` (należy się upewnić, że skrypty powershell są włączone w systemie) z folderu repozytorium:

```
.\setup.ps1
```

Po pierwszym wykonaniu tego skryptu nie trzeba już go więcej wywoływać chyba, że nastąpi manualna ingerencja w stworzone środowiska lub instalacje. Kod w tym repozytorium jest wykonywany z poziomu notatników `.ipynb` w folderze `notebooks/`, więc pliki w folderze `src/` pełnią jedynie role pomocniczą i nie należy ich włączać samych. **Kod musi zostać wykonany z poziomu instancji JupyterLab stworzonej przez skrypt**:

```
.\launch_jupyter.ps1
```

**WAŻNE**: Notebooki mogą nie działać poprawnie, gdy zostaną włączone w inny sposób, co jest spowodowane tym, że ten skrypt również konfiguruje połączenie między Python i R.
