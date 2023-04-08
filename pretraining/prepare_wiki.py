import click
from joblib import cpu_count
from datasets import load_dataset

def prepare_wiki_dataset(dataset, test_size, seed):
    
    def join_title_and_body(row):
        output = {"text": []}
        for title, text in zip(row["title"], row["text"]):
            output["text"].append(title + "\n" + text)
        return output
    
    dataset = dataset.map(
        join_title_and_body,
        batched=True,
        batch_size=1_000,
        num_proc=cpu_count()
    )

    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset

@click.command()
@click.option("-i", "--id", default="20220301.de")
@click.option("-o", "--output", type=click.Path(writable=True))
@click.option("-t", "--test_size", default=0.05)
@click.option("-s", "--seed", default=42)
def main(id, output, test_size, seed):
    dataset = load_dataset("wikipedia", id)["train"]
    dataset = prepare_wiki_dataset(dataset, test_size, seed)
    dataset.save_to_disk(output)

if __name__ == "__main__":
    main()





