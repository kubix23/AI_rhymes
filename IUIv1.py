import os
import random
import re
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

from RhymesScorer.score import score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def myscore(x: str) -> float|int:
    """
    Calculates the total rhyming score of a given input string based on the `score` function in advanced mode.

    The function calls `score` with the parameters:
    - type="advanced"
    - threshold=0

    The returned nested list is flattened and all numerical values
    are summed into a single scalar.

    Parameters
    ----------
    x : str
        Input data passed directly to the `score` function.
        The expected type is string.

    Returns
    -------
    float or int
        The sum of all values returned by `score` after flattening
        the nested structure.

    Notes
    -----
    This function assumes that `score(x, type="advanced", threshold=0)`
    returns a one-level nested iterable (e.g. a list of lists)
    containing numeric values.
    """
    return sum(sum(score(x, type="advanced", threshold = 0),[]))

def mask_random_words(text: str, num_masks: int=1) -> tuple[str, list[str]]:
    """
    Randomly masks a specified number of words in the input text.

    The function tokenizes the text while preserving whitespace, randomly
    selects word positions from the first 100 eligible tokens, replaces
    them with the "[MASK]" token, and returns both the masked text and
    the original masked words.

    Parameters
    ----------
    text : str
        Input text in which words will be randomly masked.
    num_masks : int, optional
        Number of words to mask in the text (default is 1).

    Returns
    -------
    tuple[str, list[str]]
        A tuple containing:
        - masked_text : str
            The input text with selected words replaced by "[MASK]".
        - labels : list of str
            A list of the original words that were masked, in the order
            they were selected.

    Notes
    -----
    - Tokenization is performed using a regular expression split that
      preserves whitespace tokens.
    - Only tokens that are non-empty and not purely whitespace are
      considered valid words.
    - Masked positions are sampled only from the first 100 valid words.
    - The function assumes that `num_masks` does not exceed the number
      of available word tokens.

    Example
    -------
    >>> mask_random_words("This is a simple test sentence.", num_masks=2)
    ('[MASK] is a simple [MASK] sentence.', ['This', 'test'])
    """
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, tok in enumerate(tokens) if tok.strip() != "" and not tok.isspace()]
    mask_positions = random.sample(word_indices[:100], num_masks)
    labels = [tokens[i] for i in mask_positions]
    for pos in mask_positions:
        tokens[pos] = "[MASK]"
    return "".join(tokens), labels

class TextDataset(Dataset):
    """
    PyTorch Dataset for masked language modeling with optional rewards.

    This dataset handles both original and masked versions of input texts.
    It supports dynamic masking and stores associated scores/rewards for
    reinforcement learning tasks.

    Attributes
    ----------
    texts : list of str
        Original input texts.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to encode text into token IDs.
    num_masks : int
        Number of words to mask in each text.
    scoredTexts : list of float
        Scores associated with each text, used as reward signals.
    maskedText : list of tuple[str, list[str]]
        Texts with randomly masked words and the list of original masked words.
    tokenizerMaskedText : list of tuple[dict, list[str]]
        Tokenized masked texts with rewards attached.
    tokenizerText : list of dict
        Tokenized original texts.

    Parameters
    ----------
    texts : list of tuples
        Each tuple contains two elements :

            - text - The original text string.
            - score - The score or reward associated with the text.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to encode text into token IDs.
    num_masks : int, optional
        Number of words to mask in each text (default is 1).

    Methods
    -------
    __len__()
        Returns the number of texts in the dataset.
    __getitem__(idx)
        Returns a dictionary containing:
        - "tokens": tokenized original text
        - "tokens_masked": tokenized masked text with reward attached

    Notes
    -----
    - The masked text is created using `mask_random_words`.
    - Rewards are stored as PyTorch tensors on the global `device`.
    - Tokenized sequences are padded/truncated to a maximum length of 256.
    - The class prints progress messages when creating masked and tokenized data.
    """
    def __init__(self, texts, tokenizer, num_masks=1):
        self.texts = [i[0] for i in texts]
        self.tokenizer = tokenizer
        self.num_masks = num_masks
        self.scoredTexts = [float(i[1]) for i in texts]
        self.maskedText = [mask_random_words(self.texts[idx],self.num_masks) for idx in range(len(self.texts))]
        print("maskedText created")
        self.tokenizerMaskedText = [
            (self.tokenizer(
                self.maskedText[idx][0],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=256
            ),
             self.maskedText[idx][1]
            )
            for idx in range(len(self.maskedText))
        ]
        print("tokenizerMaskedText created")
        self.tokenizerText = [
            self.tokenizer(
                self.texts[idx],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=256
            )
            for idx in range(len(self.maskedText))
        ]
        print("tokenizerText created")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizerText[idx]
        tokens_masked = self.tokenizerMaskedText[idx]
        tokens_masked[0]["reward"] = torch.tensor(self.scoredTexts[idx], dtype=torch.float, device=device)
        return {
            "tokens": tokens,
            "tokens_masked": tokens_masked,
        }

def mask_tokens(inputs, tokenizer, mlm_probability: float=0.15):
    """
   Prepare masked tokens and corresponding labels for masked language modeling (MLM).

   This function randomly selects input tokens to be masked according to
   `mlm_probability`, applies the standard BERT-style masking strategy,
   and produces labels suitable for loss computation.

   Masking strategy:
   - With probability `mlm_probability`, a token is selected for masking.
   - Of the selected tokens:
       * 80% are replaced with the tokenizer's mask token.
       * 10% are replaced with a random token.
       * 10% remain unchanged.
   - Tokens that are not selected for masking have their labels set to -100
     so they are ignored by the loss function.

   Parameters
   ----------
   inputs : torch.Tensor
       Tensor of token IDs with shape `(batch_size, sequence_length)`.
       This tensor is modified in-place and returned as masked inputs.
   tokenizer : transformers.PreTrainedTokenizer
       Tokenizer used to identify special tokens and the mask token ID.
   mlm_probability : float, optional
       Probability of masking each token (default is 0.15).

   Returns
   -------
   tuple[torch.Tensor, torch.Tensor]
       A tuple containing:
       - masked_inputs : torch.Tensor
           Input tensor with tokens masked according to the MLM strategy.
       - labels : torch.Tensor
           Tensor of the same shape as `inputs`, where masked token positions
           contain the original token IDs and unmasked positions are set to -100.

   Notes
   -----
   - Special tokens (e.g., [CLS], [SEP], [PAD]) are never masked.
   - The returned `labels` tensor is intended to be used with
     `torch.nn.CrossEntropyLoss(ignore_index=-100)`.
   - This implementation follows the original BERT masking procedure.
   """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def train(model, dataloader, optimizer, criterion, tokenizer, epochs: int):
    """
    Train a masked language model using a PyTorch training loop.

    This function performs supervised training for a masked language
    modeling (MLM) task. For each batch, input tokens are dynamically
    masked, passed through the model, and optimized using the provided
    loss function.

    Parameters
    ----------
    model : torch.nn.Module
        A transformer-based language model that returns logits in
        `outputs.logits`.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches containing tokenized inputs under
        the key `"tokens"`.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : torch.nn.Module
        Loss function, typically `torch.nn.CrossEntropyLoss` with
        `ignore_index=-100`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for masking tokens and determining vocabulary size.
    epochs : int
        Number of training epochs.

    Returns
    -------
    None
        The function trains the model in-place and prints the average
        loss after each epoch.

    Notes
    -----
    - The model is set to training mode via `model.train()`.
    - Token masking is performed dynamically for each batch using
      `mask_tokens`.
    - The loss is computed only on masked token positions.
    - Gradients are cleared before backpropagation and updated once per
      batch.
    - The function assumes the existence of a global `device` variable
      (CPU or CUDA).

    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = batch["tokens"]['input_ids'].squeeze(1).to(device)
            attention_mask = batch["tokens"]['attention_mask'].squeeze(1).to(device)
            inputs, labels = mask_tokens(inputs, tokenizer)
            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits.view(-1, len(tokenizer)), labels.view(-1))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def reinforceTrain(model, dataloader, tokenizer, optimizer, epochs: int = 10):
    """
    Train a masked language model using a REINFORCE-style policy gradient algorithm.

    This function treats masked token prediction as a stochastic policy.
    For each masked position, the model samples tokens from the predicted
    probability distribution, reconstructs the full sequence, and evaluates
    it using an external reward function (`myscore`). The model parameters
    are optimized to maximize the expected reward.

    Parameters
    ----------
    model : torch.nn.Module
        Transformer-based language model returning logits for each token.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches with masked tokenized inputs under
        the key `"tokens_masked"`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for decoding sequences and identifying mask tokens.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    epochs : int, optional
        Number of training epochs (default is 10).

    Returns
    -------
    None
        The function trains the model in-place and prints the average
        training loss and reward per epoch.

    Notes
    -----
    - The model is trained in stochastic mode using sampled actions rather
      than greedy decoding.
    - Masked token positions are treated as independent categorical
      distributions.
    - The reward signal is provided by the external `myscore` function.
    - Advantage normalization is applied by subtracting the batch mean
      reward to reduce variance.
    - Entropy regularization is used to encourage exploration.
    - The loss is normalized by the number of masked tokens per sequence.
    - The function assumes a global `device` variable.

    Training Objective
    ------------------
    The optimized objective is:

        L = -E[(log π(a|s) / M) * (R - E[R]) + λ * H(π)]

    where:
    - π(a|s) is the policy induced by the language model,
    - M is the number of masked tokens,
    - R is the reward returned by `myscore`,
    - H(π) is the policy entropy,
    - λ is the entropy coefficient (0.05).
    """
    model.train()
    for epoch in range(epochs):
        loss_list = []
        reward_list = []
        for prompt in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = prompt["tokens_masked"][0]['input_ids'].squeeze(1).to(device)
            attention_mask = prompt["tokens_masked"][0]['attention_mask'].squeeze(1).to(device)

            outputs = model(inputs, attention_mask=attention_mask).logits.view(-1,len(tokenizer))

            mask_token_index = inputs.view(-1) == tokenizer.mask_token_id
            masked_logits = outputs[mask_token_index]

            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            actions = dist.sample()

            new_inputs = inputs.clone()
            new_inputs.view(-1)[mask_token_index] = actions
            decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_inputs.view(inputs.shape)]
            rewards = torch.tensor(
                [myscore(text) for text in decoded],
                device=device
            )

            log_probs = dist.log_prob(actions).view(inputs.shape[0], -1).sum(dim=1)
            entropy = dist.entropy().view(inputs.shape[0], -1).sum(dim=1)
            advantage = (rewards - rewards.mean().detach())
            num_masks = mask_token_index.view(inputs.shape[0], -1).sum(dim=1).clamp(min=1)
            policy_loss = log_probs / num_masks
            loss = -(
                    policy_loss * advantage
                    + 0.05 * entropy
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reward_list.append(rewards.mean())
            loss_list.append(loss)

        print(f"Train loss: {sum(loss_list) / len(dataloader)} Reward: {sum(reward_list) / len(dataloader)}")

def evaluate(model, dataloader, criterion, tokenizer):
    """
    Evaluate a masked language model on a validation or test dataset.

    This function computes the average masked language modeling (MLM)
    loss over the provided dataset without updating model parameters.
    Token masking is applied dynamically for each batch.

    Parameters
    ----------
    model : torch.nn.Module
        Transformer-based language model returning logits in
        `outputs.logits`.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches containing tokenized inputs under
        the key `"tokens"`.
    criterion : torch.nn.Module
        Loss function, typically `torch.nn.CrossEntropyLoss` with
        `ignore_index=-100`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for masking tokens and determining vocabulary size.

    Returns
    -------
    float
        Average MLM loss over the entire dataset.

    Notes
    -----
    - The model is switched to evaluation mode using `model.eval()`.
    - Gradient computation is disabled via `torch.no_grad()` to reduce
      memory usage and improve evaluation speed.
    - Token masking is performed dynamically using `mask_tokens`, meaning
      evaluation results may vary slightly between runs.
    - The function assumes the existence of a global `device` variable.

    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["tokens"]['input_ids'].squeeze(1).to(device)
            attention_mask = batch["tokens"]['attention_mask'].squeeze(1).to(device)
            inputs, labels = mask_tokens(inputs, tokenizer)
            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits.view(-1, len(tokenizer)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def reinforceEvaluate(model, dataloader, tokenizer):
    """
    Evaluate a masked language model using a REINFORCE-style policy.

    This function evaluates the model on a dataset without updating
    parameters. For each masked position, actions are sampled from
    the model's predicted probability distribution, sequences are
    decoded, and rewards are computed using the `myscore` function.
    The policy loss is also computed but not used for optimization.

    Parameters
    ----------
    model : torch.nn.Module
        Transformer-based language model returning logits for each token.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches with masked tokenized inputs under
        the key `"tokens_masked"`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for decoding sequences and identifying mask tokens.

    Returns
    -------
    tuple
        A tuple containing:
        - avg_loss : torch.Tensor
            Average REINFORCE-style loss across all batches.
        - total_rewards : list[torch.Tensor]
            List of reference reward tensors for each batch.
        - total_rewards_model : list[torch.Tensor]
            List of reward tensors obtained from model-generated sequences
            for each batch.

    Notes
    -----
    - The model is switched to evaluation mode using `model.eval()`.
    - Gradients are disabled using `torch.no_grad()`.
    - Masked positions are sampled stochastically using the model's
      predicted categorical distributions.
    - The function assumes the existence of a global `device` variable.
    - The loss is calculated as:
        L = -Σ log π(a|s) * (R_model - R_reference)
      where π(a|s) is the policy probability and R_model / R_reference
      are rewards for model-generated and reference sequences, respectively.
    """

    model.eval()
    total_loss = 0
    total_rewards = []
    total_rewards_model = []
    with torch.no_grad():
        for prompt in tqdm(dataloader, desc="Evaluating"):
            inputs = prompt["tokens_masked"][0]['input_ids'].squeeze(1).to(device)
            attention_mask = prompt["tokens_masked"][0]['attention_mask'].squeeze(1).to(device)

            outputs = model(inputs, attention_mask=attention_mask).logits.view(-1,len(tokenizer))

            mask_token_index = inputs.view(-1) == tokenizer.mask_token_id
            masked_logits = outputs[mask_token_index]

            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            new_inputs = inputs.clone()
            new_inputs.view(-1)[mask_token_index] = actions
            decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_inputs.view(inputs.shape)]
            rewards = torch.tensor(
                [myscore(text) for text in decoded],
                dtype=torch.float32,
                device=device
            )
            loss = -(
                    log_probs.view(inputs.shape[0], -1).sum(dim=1) *
                    (rewards - prompt["tokens_masked"][0]["reward"])
            ).mean()
            total_loss += loss
            total_rewards.append(prompt["tokens_masked"][0]["reward"])
            total_rewards_model.append(rewards)
    return total_loss / len(dataloader), total_rewards, total_rewards_model

def loadModelTokenizer(path:str, new: bool=True):
    """
    Load a pre-trained BERT masked language model and its tokenizer.

    This function loads a BERT model for masked language modeling (MLM)
    and its corresponding tokenizer from the specified path. If `new` is
    True, it extends the tokenizer with any missing special tokens and
    resizes the model embeddings accordingly.

    Parameters
    ----------
    path : str
        Path to the pre-trained model and tokenizer.
    new : bool, optional
        If True, extend the tokenizer with missing special tokens and
        resize the model embeddings (default is True). If False, the
        model and tokenizer are loaded as-is.

    Returns
    -------
    tuple
        A tuple containing:
        - model : transformers.BertForMaskedLM
            The loaded BERT model moved to the global `device`.
        - tokenizer : transformers.BertTokenizer
            The corresponding tokenizer, possibly extended with new tokens.

    Notes
    -----
    - The function assumes a global `device` variable specifying CPU or
      GPU.
    - If `new=True`, any tokens in `{'\\n'}` that are missing from the
      tokenizer vocabulary will be added, and the model's embedding
      layer is resized to match the updated tokenizer size.
    - A message "Tokenizer loaded, Model Loaded" is printed upon successful loading.

    """

    if new:
        model = BertForMaskedLM.from_pretrained(path).to(device)
        tokenizer = BertTokenizer.from_pretrained(path)
        token = {'\n'} - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(token))
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = BertForMaskedLM.from_pretrained(path).to(device)
        tokenizer = BertTokenizer.from_pretrained(path)
    print("Tokenizer loaded, Model Loaded")
    return model, tokenizer

def loadPoems(path_scores_poems:str, path_dir_polish_poetry: str, load_num: int = None) -> list[tuple[str, str]]:
    """
    Load Polish poems and their associated scores from specified paths.

    This function reads poem text files from a directory and their
    corresponding scores from a score file. Poems are filtered to include
    only those with more than 5 non-empty lines and truncated to a maximum
    of 40 lines. Optionally, the number of poems loaded can be limited.

    Parameters
    ----------
    path_scores_poems : str
        Path to the file containing poem scores, with each line in the
        format "filename|score".
    path_dir_polish_poetry : str
        Directory path containing the poem text files.
    load_num : int, optional
        Maximum number of poem files to load. If None, all available
        poems in the directory are considered (default is None).

    Returns
    -------
    list of tuple[str, str]
        A list of tuples, where each tuple contains:
        - poem_text : str
            Concatenated text of the poem, with lines joined by " \n".
        - score : str
            Corresponding score from the scores file.

    Notes
    -----
    - Only poems present in the scores file are loaded.
    - Each poem is truncated to at most 40 non-empty lines.
    - Poems with 5 or fewer non-empty lines are skipped.
    - A message indicating the total number of loaded poems is printed.

    """

    text_list = []
    with open(path_scores_poems, "r", encoding="utf-8") as f:
        poetry_list = {i.split('|')[0]:i.split('|')[1].rstrip() for i in f.readlines() }

    all_poems_list = os.listdir(path_dir_polish_poetry)
    if load_num is None:
        load_num = len(all_poems_list)

    for file in all_poems_list[:load_num]:
        if file in poetry_list:
            with open(path_dir_polish_poetry + file, "r", encoding="utf-8") as f:
                text = f.readlines()
                table = [i.strip() for i in text if i != '\n'][:40]
                if len(table) > 5:
                    text_list.append((' \n'.join(table), poetry_list[file]))
    print(f"{len(text_list)} poems loaded")
    return text_list

def loadRhymes(path_scores_rhymes:str, path_rhymes: str, load_num: int = None) -> list[tuple[str, str]]:
    """
    Load rhyming sequences and their associated scores from specified files.

    This function reads rhymes from a text file and their corresponding
    scores from a separate CSV-like file. Only entries present in the
    scores file are loaded. Optionally, the number of rhymes to load
    can be limited.

    Parameters
    ----------
    path_scores_rhymes : str
        Path to the file containing rhyme scores, with each line in the
        format "index,score".
    path_rhymes : str
        Path to the text file containing rhymes, one per line.
    load_num : int, optional
        Maximum number of rhymes to load from the rhymes file.
        If None, all rhymes are considered (default is None).

    Returns
    -------
    list of tuple[str, str]
        A list of tuples, where each tuple contains:
        - rhyme_text : str
            The text of the rhyme, with "|" characters replaced by spaces.
        - score : str
            The corresponding score from the scores file.

    Notes
    -----
    - Only rhymes with indices present in the scores file are loaded.
    - If `load_num` is specified, only the first `load_num` lines from
      the rhymes file are considered.
    - A message indicating the total number of loaded rhymes is printed.
    """
    text_list = []
    with open(path_scores_rhymes, "r", encoding="utf-8") as f:
        rhymes_list = {i.split(',')[0]: i.split(',')[1].rstrip() for i in f.readlines()}

    with open(path_rhymes, "r", encoding="utf-8") as f:
        text = f.readlines()
        if load_num is not None:
            text = text[:load_num]

        text = [(i.replace('|', ' ').strip(), rhymes_list[str(num)]) for num, i in enumerate(text) if
                str(num) in rhymes_list]
        if text:
            text_list.extend(text)
    print(f"{len(text_list)} rhymes loaded")
    return text_list

def main():
    model, tokenizer = loadModelTokenizer("dkleczek/bert-base-polish-cased-v1", new=True)
    # model, tokenizer = loadModelTokenizer("./mlm_pretrained_model_from_scratch", new=False)

    # text_list = loadPoems(
    #     "E:/Projects/IUI/scoresPoetry",
    #     "E:/Projects/IUI/Polish_poetry/",
    #     700
    # )

    text_list = loadRhymes(
        "E:/Projects/IUI/scoresRhymes",
        "E:/Projects/IUI/rhymes.csv",
        100
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    criterion = nn.CrossEntropyLoss()
    dataset = TextDataset(text_list, tokenizer, 1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    train(model, dataloader, optimizer, criterion, tokenizer, epochs=18)
    eval_loss = evaluate(model, dataloader, criterion, tokenizer)
    print(f"Evaluation Loss: {eval_loss}")

    reinforceTrain(model, dataloader, tokenizer, optimizer,epochs=2000)
    eval_loss, eval_rewards, eval_rewards_model = reinforceEvaluate(model, dataloader, tokenizer)
    print(f"Evaluation Loss: {eval_loss}")
    pprint(eval_rewards)
    pprint(eval_rewards_model)

    # model.save_pretrained("./mlm_pretrained_model_from_scratch")
    # tokenizer.save_pretrained("./mlm_pretrained_model_from_scratch")

if __name__ == "__main__":
    main()
