"""ノートブックの TODO の書き忘れを見つける．

    todo_check("TODO 1")

いま実行しているセルに「...」（ここに書く，という目印）が残っていたら，日本語のメッセージで止める．
コメント（# 以降）と文字列の中の「...」は数えない．
"""
import io
import tokenize


def _has_placeholder(src):
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.OP and tok.string == "...":
                return True
    except (tokenize.TokenError, IndentationError):
        pass
    return False


def todo_check(name):
    try:
        from IPython import get_ipython
        src = get_ipython().history_manager.input_hist_raw[-1]   # いま実行しているセル
    except Exception:
        return   # ノートブックの外では何もしない
    if _has_placeholder(src):
        raise RuntimeError(f"{name} が未記入です：「...」を消して式を書いてください")


def checkpoint(answer, name, min_chars=10):
    """✋ 止まる地点：グループで決めた答えが書かれていなければ止める．

        CP1 = "..."
        checkpoint(CP1, "止まる 1")
    """
    text = (answer or "").strip()
    if len(text) < min_chars:
        raise RuntimeError(f"{name}：グループで話し合った答えを {min_chars} 文字以上で書いてから進む（いまは {len(text)} 文字）")
    print(f"{name} OK：{text}")


def _quiz_key(name, answer):
    import hashlib
    return hashlib.sha256(f"{name}:{answer}".encode("utf-8")).hexdigest()[:8]


def quiz_check(items):
    """確認問題の答え合わせ．全問正解するまで先へ進めない．
    正解は記号のまま書かず，_quiz_key(問題名, 記号) の値で渡す（コードを見ても答えが分からないように）．

        quiz_check({"Q1": (Q1, "1a2b3c4d", "ヒント"), ...})
    """
    wrong = []
    for name, (answer, key, note) in items.items():
        a = (answer or "").strip()
        if not a:
            wrong.append(f"{name}：まだ答えていない")
        elif _quiz_key(name, a) != key:
            wrong.append(f"{name}：ちがう（ヒント：{note}）")
        else:
            print(f"{name}：正解．{note}")
    if wrong:
        raise RuntimeError("確認問題をもう一度：" + "　".join(wrong))
    print("全問正解．実験に進んでよい")
