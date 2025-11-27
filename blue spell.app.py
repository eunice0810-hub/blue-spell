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
def tokenize_text(text):
    return word_tokenize(text)


def is_candidate_word(tok, min_len, ignore_all_caps, ignore_title):
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
    text,
    filename,
    spell_checker,
    min_len,
    ignore_all_caps,
    ignore_title,
    custom_ignore=None,
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

    corrected_indices = []
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
        corrected_indices.append(i)
        corrected_count += 1

    # detokenize용 순수 토큰
    safe_tokens = [t if isinstance(t, str) else "" for t in tokens]
    corrected_text = detok.detokenize(safe_tokens)

    # 하이라이트용 토큰 (HTML span 감싸기)
    display_tokens = []
    corrected_set = set(corrected_indices)
    for idx, tok in enumerate(safe_tokens):
        if idx in corrected_set and tok.strip():
            display_tokens.append(f'<span class="corrected-word">{tok}</span>')
        else:
            display_tokens.append(tok)
    highlighted_html = detok.detokenize(display_tokens)

    stats = {
        "filename": filename,
        "total_tokens": len(safe_tokens),
        "candidate_count": len(candidate_words),
        "corrected_count": corrected_count,
    }
    return corrected_text, highlighted_html, stats


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(
        page_title="Blue Spell (Yonsei Edition)",
        layout="wide",
    )

    # ---- Custom CSS ----
    st.markdown(
        """
        <style>
        /* 전체 페이지 여백 조금 줄이기 */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* 헤더 박스 */
        .main-header {
            background: linear-gradient(90deg, #003b8e, #2563eb);
            padding: 1.6rem 2.0rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.8rem;
        }

        .main-header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
        }

        .main-header p {
            margin: 0;
            font-size: 0.95rem;
            opacity: 0.95;
        }

        /* 업로드 카드 */
        .upload-card {
            background-color: #f3f6ff;
            border: 1px solid #c7d2ff;
            border-radius: 14px;
            padding: 1.4rem 1.6rem 1.6rem 1.6rem;
            margin-bottom: 1.2rem;
        }

        .upload-card h3 {
            margin-top: 0;
            margin-bottom: 0.4rem;
        }

        .upload-card p {
            margin-top: 0;
            font-size: 0.9rem;
            color: #4b5563;
        }

        /* 교정 단어 하이라이트 */
        .corrected-word {
            background-color: #e0ecff;
            color: #003b8e;
            font-weight: 600;
            padding: 0 2px;
            border-radius: 3px;
        }

        .corrected-text-box {
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            padding: 0.9rem 1.0rem;
            background-color: #ffffff;
            font-size: 0.95rem;
            line-height: 1.6;
            max-height: 350px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- 헤더 (로고 + 타이틀) ----
    st.markdown(
        """
        <div class="main-header">
          <div style="display:flex; align-items:center; gap:1.0rem;">
            <!-- 연세대 로고: 같은 폴더에 yonsei_logo.png 파일을 넣어 두세요 -->
            <img src="yonsei_logo.png" alt="Yonsei Logo" width="46" style="border-radius: 8px; background-color:white; padding:4px;">
            <div>
              <h1>Blue Spell (Yonsei Edition)</h1>
              <p>영어 텍스트 철자 교정 및 통계 분석 도구 · 여러 개의 .txt 파일을 한 번에 처리합니다.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_nltk()
    spell = SpellChecker(language="en")

    # ---- Sidebar: 옵션 ----
    with st.sidebar:
        st.header("옵션")

        min_len = st.number_input(
            "철자 후보로 볼 최소 단어 길이",
            min_value=1,
            max_value=20,
            value=3,
        )
        ignore_all_caps = st.checkbox(
            "모두 대문자 단어 무시 (예: ABC)",
            value=True,
        )
        ignore_title = st.checkbox(
            "첫 글자만 대문자인 단어 무시 (예: Yonsei)",
            value=True,
        )

        st.markdown("---")
        st.caption("커스텀 무시 단어 목록 (.txt, 한 줄당 한 단어)")
        custom_ignore_file = st.file_uploader(
            "무시할 단어 리스트 업로드",
            type=["txt"],
            key="ignore_list",
        )

    # ---- 메인 영역: 업로드 카드 ----
    st.markdown(
        """
        <div class="upload-card">
          <h3>1. 철자 교정을 할 파일을 업로드하세요</h3>
          <p>
            • <b>.txt</b> 파일을 하나 이상 선택할 수 있습니다.<br>
            • 각 파일의 내용에 대해 철자 오류를 교정하고, 교정 통계를 제공합니다.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "텍스트 파일(.txt)을 드래그하거나 'Browse files' 버튼을 눌러 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
    )

    run_button = st.button("🔍 철자 검사 실행")

    if run_button:
        if not uploaded_files:
            st.warning("먼저 하나 이상의 .txt 파일을 업로드해 주세요.")
            return

        # 커스텀 ignore 리스트 읽기
        custom_ignore = set()
        if custom_ignore_file is not None:
            try:
                content = custom_ignore_file.read().decode("utf-8", errors="ignore")
                custom_ignore = {
                    line.strip().lower()
                    for line in content.splitlines()
                    if line.strip()
                }
                st.sidebar.success(f"무시할 단어 {len(custom_ignore)}개 로드됨")
            except Exception as e:
                st.sidebar.error(f"무시 단어 리스트 로드 실패: {e}")

        summary_rows = []

        st.markdown("### 2. 교정 결과")

        for file in uploaded_files:
            try:
                raw = file.read().decode("utf-8", errors="ignore")
            except Exception:
                raw = file.read().decode("cp949", errors="ignore")

            corrected_text, highlighted_html, stats = run_spellcheck_on_text(
                raw,
                filename=file.name,
                spell_checker=spell,
                min_len=min_len,
                ignore_all_caps=ignore_all_caps,
                ignore_title=ignore_title,
                custom_ignore=custom_ignore,
            )

            st.subheader(f"📄 파일: {file.name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("전체 토큰 수", stats["total_tokens"])
            col2.metric("철자 후보 단어 수", stats["candidate_count"])
            col3.metric("실제 교정된 단어 수", stats["corrected_count"])

            st.markdown("**교정된 텍스트 (교정된 단어는 파란색으로 표시됩니다)**")
            st.markdown(
                f'<div class="corrected-text-box">{highlighted_html}</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                label="💾 교정된 텍스트 파일 다운로드",
                data=corrected_text.encode("utf-8"),
                file_name=f"{file.name.rsplit('.', 1)[0]}_corrected.txt",
                mime="text/plain",
            )

            summary_rows.append(stats)

        if summary_rows:
            st.markdown("### 3. 전체 파일 요약 통계")
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📊 요약 통계 CSV 다운로드",
                data=csv_bytes,
                file_name="spelling_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

