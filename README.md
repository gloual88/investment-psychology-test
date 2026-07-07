# 투자 심리검사 (Investment Psychology Test)

블라인드 종목 10개(박스권 5 + 추세장 5)를 10주간 직접 매매해 보고, 그 거래내역으로
**5대 행동편향**을 진단하는 Streamlit 웹앱입니다. 국면별 처분효과·성과를 규율매매/바이앤홀드와
비교하고, 맞춤 솔루션과 진단 리포트(.docx)를 제공합니다.

## 온라인 데모
Streamlit Community Cloud로 배포하면 누구나 브라우저로 접속할 수 있습니다.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
Windows(venv 사용 시):
```powershell
..\..\pykrx_venv\Scripts\python.exe -m streamlit run app.py --server.port 8533
```

## 구성
| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 대시보드(진입점) |
| `profiler.py` | 5대 행동편향 진단·점수화 |
| `shadow_backtesting.py` | 국면별 그림자 백테스트 |
| `disposition_effect.py` | 처분효과(PGR−PLR) 계산 |
| `report.py` | 진단 리포트(.docx) 생성 |
| `prices_lab.csv`, `meta_lab.csv` | 블라인드 실험용 가격·메타(런타임 입력) |
| `prep_data.py` | 실험 데이터 재생성(yfinance, 로컬 전용·배포 불필요) |

## 배포 메모
- 차트 한글 폰트는 실행 환경에 맞춰 자동 선택(로컬 Malgun / 클라우드 NanumGothic).
  클라우드 폰트는 `packages.txt`의 `fonts-nanum`로 설치됩니다.
- 런타임에는 외부 네트워크·API 키가 필요 없습니다(CSV만 사용).
