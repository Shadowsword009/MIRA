
from langchain_ollama import ChatOllama


_active_model = "qwen2.5:1.5b"
_llm = ChatOllama(
    model=_active_model,
    temperature=0.1,
    num_ctx=4096
)


def get_llm():
    return _llm


def get_active_model():
    return _active_model


def set_model(model_name):
    global _active_model, _llm

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("A valid model name is required.")

    _llm = ChatOllama(
        model=model_name,
        temperature=0.1,
        num_ctx=4096
    )

    _active_model = model_name


def test_llm():
    response = get_llm().invoke(
        "Explain in simple terms why training a classification "
        "model with only one target class is problematic."
    )

    return response.content


if __name__ == "__main__":
    print(test_llm())
