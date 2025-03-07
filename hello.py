from typing import Iterable, List
import dspy
from pydantic import BaseModel


class CategoryType(BaseModel):
    category: str
    description: str


class ClassifySignature(dspy.Signature):
    """Given available categories, categorize `text` to one of them"""

    categories: List[CategoryType] = dspy.InputField(description="available categories")
    text: str = dspy.InputField(description="text to categorize")
    category: str = dspy.OutputField(
        description="category of text based on available categories"
    )


class ClassifierModule(dspy.Module):
    def __init__(self, categories) -> None:
        super().__init__()
        self.categories = categories
        self.pred = dspy.Predict(ClassifySignature)

    def forward(self, text: str) -> str:
        prediction = self.pred(categories=self.categories, text=text)
        return prediction.category


def validate_category(example, prediction, trace=None):
    return prediction == example.category


def optimize_for_categories(classify, trainset, **kwargs):
    tp = dspy.MIPROv2(metric=validate_category, auto="light")
    optimized_classify = tp.compile(classify, trainset=trainset, **kwargs)
    return optimized_classify


def build_trainset(text_category_zipped: Iterable):
    return [
        dspy.Example(
            text=text,
            category=category,
        ).with_inputs("text")
        for text, category in text_category_zipped
    ]


if __name__ == "__main__":
    gpt_4o_mini = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=gpt_4o_mini)

    print("Starting...")
    categories = [
        CategoryType(**{
            "category": "misc",
            "description": "No category matches this text",
        }),
        CategoryType(**{
            "category": "ok",
            "description": "there was nothing wrong",
        }),
        CategoryType(**{
            "category": "missing values",
            "description": "There was something mising in the document",
        }),
    ]
    classify = ClassifierModule(categories)
    x1 =  "cat sat on a mat"
    cat = classify(x1)
    print(cat)
    x2 =  "nothing to declare"
    cat = classify(x2)
    print(cat)
    x3 =  "missing declaration"
    cat = classify(x3)
    print(cat)

    trainset = build_trainset([(x1, "misc"), (x2, "ok"), (x3, "missing values"),])
    optimized = optimize_for_categories(classify, trainset, requires_permission_to_run=False, provide_traceback=True)

    classify = optimized
    x1 =  "cat sat on a mat"
    cat = classify(x1)
    print(cat)
    x2 =  "nothing to declare"
    cat = classify(x2)
    print(cat)
    x3 =  "missing declaration"
    cat = classify(x3)
    print(cat)

    
