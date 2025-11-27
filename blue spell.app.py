import streamlit as st
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

# ----------------------------
# NLTK setup
# ----------------------------
def ensure_nltk():
    """Ensure required NLTK resources exist (download once if missing)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


# ----------------------------
# Helpers
# ----------------------------
def tokenize_text(text: str):
    return word_tokenize(text)


def is_candidate_word(tok: str, min_len: int, ignore_all_caps: bool, ignore_title: bool) -> bool:
    if not isinstance(tok, str):
        return False
    if not tok.isalpha():
        return False
    if len(tok) < min_len:
        return False
    if ignore_all_caps and tok.isupper():
        return False
    if ignore_title and tok.istitle():
        return False
    return True


def run_spellcheck_on_text(
    text: str,
    filename: str,
    spell_checker: SpellChecker,
    min_len: int,
    ignore_all_caps: bool,
    ignore_title: bool,
    custom_ignore: set | None = None,
):
    detok = TreebankWordDetokenizer()
    tokens = tokenize_text(text)

    candidate_indices = []
    candidate_words = []

    for i, tok in enumerate(tokens):
        if is_candidate_word(tok, min_len, ignore_all_caps, ignore_title):
            lw = tok.lower()
            if custom_ignore and lw in custom_ignore:
                continue
            candidate_indices.append(i)
            candidate_words.append(lw)

    misspelled = spell_checker.unknown(candidate_words)

    corrected_count = 0
    for i in candidate_indices:
        orig = tokens[i]
        lw = orig.lower()
        if lw not in misspelled:
            continue

        suggestion = spell_checker.correction(lw)
        if not suggestion or suggestion == lw:
            continue

        # 원래 대소문자 형태 최대한 유지
        if orig.istitle():
            suggestion = suggestion.capitalize()
        elif orig.isupper():
            suggestion = suggestion.upper()

        tokens[i] = suggestion
        corrected_count += 1

    tokens = [t if isinstance(t, str) else "" for t in tokens]
    corrected_text = detok.detokenize(tokens)

    stats = {
        "filename": filename,
        "total_tokens": len(tokens),
        "candidate_count": len(candidate_words),
        "corrected_count": corrected_count,
    }
    return corrected_text, stats


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Blue Spell Yonsei (Streamlit)", layout="wide")

    st.title("🟦 Blue Spell (Yonsei Edition) – Streamlit 버전")
    st.write(
        "여러 개의 `.txt` 파일을 업로드하면, "
        "철자 오류를 교정하고 요약 통계를 보여주는 웹 앱입니다."
    )

    ensure_nltk()
    spell = SpellChecker(language="en")

    # ---- Sidebar: 옵션 ----
    with st.sidebar:
        st.header("옵션")
        min_len = st.number_input("최소 단어 길이", min_value=1, max_value=20, value=3)
        ignore_all_caps = st.checkbox("모두 대문자 단어 무시 (ABC)", value=True)
        ignore_title = st.checkbox("첫 글자만 대문자(Title Case) 무시 (e.g., Yonsei)", value=True)

        st.markdown("---")
        custom_ignore_file = st.file_uploader(
            "커스텀 ignore 리스트 (.txt, 한 줄당 한 단어)", type=["txt"], key="ignore_list"
        )

    uploaded_files = st.file_uploader(
        "철자 검사할 `.txt` 파일을 업로드하세요 (여러 개 가능)",
        type=["txt"],
        accept_multiple_files=True,
    )

    run_button = st.button("철자 검사 실행")

    if run_button:
        if not uploaded_files:
            st.warning("먼저 `.txt` 파일을 하나 이상 업로드하세요.")
            return

        # 커스텀 ignore 리스트 읽기
        custom_ignore: set[str] = set()
        if custom_ignore_file is not None:
            try:
                content = custom_ignore_file.read().decode("utf-8", errors="ignore")
                custom_ignore = {
                    line.strip().lower()
                    for line in content.splitlines()
                    if line.strip()
                }
                st.sidebar.success(f"Ignore 단어 {len(custom_ignore)}개 로드됨")
            except Exception as e:
                st.sidebar.error(f"Ignore 리스트 로드 실패: {e}")

        summary_rows = []

        for file in uploaded_files:
            try:
                raw = file.read().decode("utf-8", errors="ignore")
            except Exception:
                raw = file.read().decode("cp949", errors="ignore")

            corrected_text, stats = run_spellcheck_on_text(
                raw,
                filename=file.name,
                spell_checker=spell,
                min_len=min_len,
                ignore_all_caps=ignore_all_caps,
                ignore_title=ignore_title,
                custom_ignore=custom_ignore,
            )

            st.subheader(f"파일: {file.name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("전체 토큰 수", stats["total_tokens"])
            col2.metric("철자 후보 단어 수", stats["candidate_count"])
            col3.metric("교정된 단어 수", stats["corrected_count"])

            st.text_area(
                "교정된 텍스트 미리보기",
                corrected_text[:3000],
                height=200,
            )

            st.download_button(
                label="교정된 파일 다운로드",
                data=corrected_text.encode("utf-8"),
                file_name=f"{file.name.rsplit('.', 1)[0]}_corrected.txt",
                mime="text/plain",
            )

            summary_rows.append(stats)

        if summary_rows:
            st.markdown("### 전체 파일 요약")
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="요약 CSV 다운로드",
                data=csv_bytes,
                file_name="spelling_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
