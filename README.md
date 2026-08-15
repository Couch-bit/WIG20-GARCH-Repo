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

## Dane

Wykorzystane dane znajdują w folderze `data/stocks/` w postaci plików `.csv` nazwanych według tickerów giełdowych na GPW na stan z dnia 29 lipca 2026 (lub z ostatniego dnia w którym dana spółka była notowana na giełdzie). Dane zostały pobrane z serwisu [investing.com](https://www.investing.com) w dniu 29 lipca 2026 (w godzinach popołudniowych po końcu handlu i aktualizacji danych na stronie). Pliki zawierają dane od początku roku 2014 do 29 lipca 2026 z pominięciem dni handlowych dla których brakuje danych dla danej spółki (nie była handlowana). Dokładniej, zawierają one przede wszystkim skorygowane ceny zamknięcia obliczone dla:
- [Alior Bank SA (ALR)](https://www.investing.com/equities/alior-bank-historical-data),
- [CD PROJEKT SA (CDR)](https://www.investing.com/equities/cdproject-historical-data),
- [Cyfrowy Polsat SA (CPS)](https://www.investing.com/equities/cyfrowy-polsat-sa-historical-data),
- [Dino Polska SA (DNP)](https://www.investing.com/equities/dino-polska-sa-historical-data),
- [Erste Bank Polska SA (EBP)](https://www.investing.com/equities/bz-wbk-historical-data),
- [Jastrzebska Spotka Weglowa SA (JSW)](https://www.investing.com/equities/jastrzebska-spolka-weglowa-historical-data),
- [KGHM Polska Miedz SA (KGH)](https://www.investing.com/equities/kghm-polska-miedz-sa-historical-data),
- [LPP SA (LPP)](https://www.investing.com/equities/lpp-historical-data),
- [Grupa Lotos SA (LTS)](https://investing.com/equities/grupa-lotos-sa-historical-data),
- [mBank SA (MBK)](https://www.investing.com/equities/bre-bank-sa-historical-data),
- [MODIVO SA (MDV)](https://www.investing.com/equities/ccc-historical-data),
- [Orange Polska SA (OPL)](https://www.investing.com/equities/tpsa-historical-data),
- [Bank Polska Kasa Opieki SA (PEO)](https://www.investing.com/equities/bank-pekao-sa-historical-data),
- [PGE Polska Grupa Energetyczna SA (PGE)](https://www.investing.com/equities/pge-polska-historical-data),
- [Polskie Gornictwo Naftowe i Gazownictwo SA (PGN)](https://www.investing.com/equities/gornictwo-naftowe-gazownictwo-historical-data),
- [Polski Koncern Naftowy ORLEN SA (PKN)](https://www.investing.com/equities/pkn-orlen-historical-data),
- [Powszechna Kasa Oszczednosci Bank Polski SA (PKO)](https://www.investing.com/equities/pko-bank-polski-historical-data),
- [Play Communications SA (PLY)](https://www.investing.com/equities/play-communications-historical-data),
- [PZU SA (PZU)](https://www.investing.com/equities/pzu-historical-data),
- [Tauron Polska Energia SA (TPE)](https://www.investing.com/equities/tauron-polska-energia-historical-data).

Dodatkowo pobrano miesięczne stopy zwrotu dla 10-letnich obligacji rządowych z notowań serwisu [stooq.pl](https://stooq.pl/). Stopa zwrotu wolna od ryzyka na następny miesiąc została przyjęta jako najniższa stopa zwrotu z poprzedniego miesiąca według tych danych. Dokładniej, wykorzystano miesięczne dane od początku roku 2014 do końca lipca 2026. Dane zostały pobrane dnia 15 sierpnia w godzinach popołudniowych z pomocą [linku](https://stooq.pl/q/d/?f=20140101&t=20260731&s=10yply.b&c=0&i=m) i są zawarte w pliku `data/10yply_b_m.csv`.
