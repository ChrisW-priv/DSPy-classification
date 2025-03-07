import dspy 


CategoryType = str | dict[str, str]

class ClassifySignature(dspy.Signature):
    """Given available categories, categorize `text` to one of them"""
    categories: list[CategoryType] = dspy.InputField(description='available categories')
    text: str = dspy.InputField(description='text to categorize')
    category: str = dspy.OutputField(description='category of text based on available categories')


class ClassifierModule(dspy.Module):
    def __init__(self, categories) -> None:
        super().__init__()
        self.categories = categories
        self.pred = dspy.Predict()

    def forward(self, text: str) -> str:
        prediction = self.pred(text)
        return prediction.category


def main(text: str):
    categories = [
        {
            'category': 'misc',
            'description': 'No category matches this text',
        },
        {
            'category': 'ok',
            'description': 'there was nothing wrong',
        },
        {
            'category': 'missing values',
            'description': 'There was something mising in the document',
        },
    ]
    cat_module = ClassifierModule(categories)
    predicted_cat = cat_module(text)
    return predicted_cat


if __name__ == "__main__":
    cat = main("cat sat on a mat")
    print(cat)
    cat = main("nothing to declare")
    print(cat)
    cat = main("missing declaration")
    print(cat)

