# 케라스 문서

케라스 문서 소스가 이 디렉터리의 `sources/` 아래에 있다.
이 문서에서는 [MkDocs](http://mkdocs.org)에서 구현하는 확장 마크다운 형식을 쓴다.

## 문서 빌드하기

- MkDocs 설치: `pip install mkdocs`
- `docs/` 폴더로 `cd` 해서 다음 실행:
    - `python autogen.py`
    - `mkdocs serve`    # 로컬 웹 서버 시작:  [localhost:8000](localhost:8000)
    - `mkdocs build`    # "site" 디렉터리에 정적 사이트 구축
