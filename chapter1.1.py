class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
         for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            logits = self.model(input_ids[None, -self.cfg.n_ctx :])
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
            
             # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
                
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

    return self.tokenizer.decode(input_ids)

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        raise NotImplementedError()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        # Applying temp means multiplying logits * 1/temperature so:
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        #The frequency penalty: count the number of occurrences of each token, 
        #then subtract freq_penalty for each occurrence.
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        top_k_logits, top_k_token_ids = logits.topk(k)
         # Get sampled token (which is an index corresponding to the list of top-k tokens)
         # Categorical accepts unnormalised logits
        sampled_token_idx = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        # Get the actual token id, as an int
        return top_k_token_ids[sampled_token_idx].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        num_to_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        num_to_keep = max(num_to_keep, min_tokens_to_keep)
        keep_idx = indices[:num_to_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()
        
    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        """
        #Tuple (logprob sum, string completion)
        List[Tuple[List[int], float]]  # (token_ids, log_prob_sum)
        initial_input_ids = tokenizer.encode(prompt)
        initial_beam = (initial_input_ids, 0.0)  # logprob = 0 at the start
        beams = [initial_beam]
        
        
        #logprobs.topk(k) for no_repeat_ngram_size
        
        raise NotImplementedError()


t.set_grad_enabled(False)  # gradients are not necessary for sampling

model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)
tokenizer = reference_gpt2.tokenizer
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Testing greedy decoding\nPrompt:   {prompt!r}")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)

print(f"Expected: {expected!r}\nActual:   {output!r}\n")
assert output == expected

print("Tests passed!")










@dataclass
class Beams:
    """Class to store beams during beam search."""

    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def __getitem__(self, batch_idx) -> "Beams":
        """Allows you to create new beams from old beams by slicing along batch dim (useful for `filter`)."""
        return Beams(
            self.model, self.tokenizer, self.logprob_sums[batch_idx], self.tokens[batch_idx]
        )

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        """
        Starting from the current set of beams (i.e. self.tokens) and returns a new set of `len(self.tokens) * k` beams,
        containing the best `k` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with a repeating n-gram
        of this length.
        """
        #self.tokens: shape (batch, seq)
        
        #self.logprob_sums: shape (batch,)
        
        logits = self.model(self.tokens)
        next_token_logits = logits[:, -1, :]
        logprobs = next_token_logits.log_softmax(dim=-1)

        topk_logprobs, topk_tokens = logprobs.topk(k, dim=-1)

        repeated_tokens = self.tokens.repeat_interleave(k, dim=0)
        topk_tokens_flat = topk_tokens.reshape(-1, 1)
        new_tokens = t.cat([repeated_tokens, topk_tokens_flat], dim=-1)

        new_logprobs = self.logprob_sums.repeat_interleave(k, dim=0) + topk_logprobs.reshape(-1)

        return Beams(self.model, self.tokenizer, new_logprobs, new_tokens)



    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        # 1. Check which beams are terminated
        is_terminated = self.tokens[:, -1] == self.tokenizer.eos_token_id

        # 2. Get indices of terminated and non-terminated beams
        terminated_indices = t.where(is_terminated)[0]
        alive_indices = t.where(~is_terminated)[0]

        # 3. Get terminated and alive beams
        terminated_beams = self[terminated_indices]
        alive_beams = self[alive_indices]

        # 4. Sort both groups by logprob_sum
        sorted_terminated = t.argsort(terminated_beams.logprob_sums, descending=True)
        sorted_alive = t.argsort(alive_beams.logprob_sums, descending=True)

        # 5. Take top-k from each
        top_terminated = terminated_beams[sorted_terminated[:k]]
        top_alive = alive_beams[sorted_alive[:k]]

        return top_alive, top_terminated



    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logprob sums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logprob sum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int | None = None,
) -> list[tuple[float, str]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
    prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
    we've generated `num_returns_sequences` terminating sequences.
    """
    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    final_logprobs_and_completions = []  # we add to this list as we get terminated beams
    best_beams = Beams(
        self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens
    )  # start with just 1 beam

    for _ in tqdm(range(max_new_tokens)):
        t.cuda.empty_cache()

        # Generate & filter beams
        best_beams = best_beams.generate(k=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
        best_beams, best_beams_terminated = best_beams.filter(k=num_beams)

        # Add terminated beams to our list, and return early if we have enough
        final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    # Return terminated beams plus the best ongoing beams of length `orig_len + max_new_tokens`
    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    return final_logprobs_and_completions[:num_return_sequences]


TransformerSampler.beam_search = beam_search



# Start with prompt "When I was", get top 3 tokens (and their logprobs), and use that to create & display the top 3 beams
prompt = "When I was"
tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
logprobs = model(tokens)[0, -1].log_softmax(-1)
top_logprobs, top_tokens = logprobs.topk(k=3, dim=-1)

new_tokens = t.concat([tokens.repeat(3, 1), top_tokens.unsqueeze(-1)], dim=-1)

beams = Beams(model, tokenizer, logprob_sums=top_logprobs, tokens=new_tokens)
beams.print()

print("Testing generate...")
new_beams = beams.generate(k=3, no_repeat_ngram_size=1)
new_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-4.8, "When I was a child"),
    (-4.9, "When I was a little"),
]

for i, (logprob_sum, completion) in enumerate(new_beams.logprobs_and_completions[:3]):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

print("All tests for `generate` passed!")


print("Testing `filter`...")

best_beams, terminated_beams = new_beams.filter(3)
best_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-3.2, "When I was growing up"),
    (-4.6, "When I was in the"),
]

for i, (logprob_sum, completion) in enumerate(best_beams.logprobs_and_completions):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

assert len(terminated_beams.logprobs_and_completions) == 0

print("All tests for `filter` passed!")


print("Testing `no_repeat_ngram_size`...")

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1)
new_beams.print(title="Completions with no ngram restriction")
assert all(
    "I was" in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "Without restriction, all beams should be completed as '...I was...'"

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1, no_repeat_ngram_size=2)
new_beams.print(title="Completions with no repeated bigrams")
assert all(
    "I was" not in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "With no repeated bigrams, no beams should contain a second '...I was...'"


sampler = TransformerSampler(model, tokenizer)

prompt = "The ships hung in the sky in much the same way that"
orig_len = len(tokenizer.encode(prompt))

final_logitsums_and_completions = sampler.beam_search(
    prompt=prompt,
    num_return_sequences=3,
    num_beams=40,
    max_new_tokens=60,
    no_repeat_ngram_size=2,
)

# Print all the best output
for logprob_sum, text in final_logitsums_and_completions:
    avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp()
    rprint(f"Avg token prob = {avg_logprob_as_prob:.3f}\nBest output:\n[bold dark_orange]{text}")
    
    
# Define a type for a single layer's cache entry (useful for type checking in later functions)
KeyValueCacheTensor = Float[Tensor, "2 batch seq_len n_heads d_head"]

class KeyValueCache(Tensor):
    '''
    This class holds tensors of key and value vectors, to be used for caching.

    If we define it using cfg and batch then it's initialized as empty, but
    we can also define it from kv_cache_entries.
    '''
    @classmethod
    def new_empty(cls, cfg: Config, batch: int = 1) -> "KeyValueCache":
        '''
        Doing a forward pass on a cache created in this way indicates "we don't
        yet have a cache, but we want this forward pass to return a cache".
        Whereas using cache=None in a forward pass indicates we don't want to
        return a cache.
        '''
        shape = (cfg.n_layers, 2, batch, 0, cfg.n_heads, cfg.d_head)
        return cls(*shape).to(device)

    # Define a handful of properties, so they can be referenced directly rather than
    # indexing (which is more likely to lead to mistakes)

    @property
    def k(self) -> Tensor:
        return self[:, 0]

    @property
    def v(self) -> Tensor:
        return self[:, 1]

    @property
    def batch(self) -> int:
        return self.shape[2]

    @property
    def seq_len(self) -> int:
        return self.shape[3]


# Example implementation:
cfg = model.cfg
batch = 6
kv_cache = KeyValueCache.new_empty(cfg, batch)

print(f"Shape of all kv-cache = {tuple(kv_cache.shape)}")
print(f"Shape of just k-cache = {tuple(kv_cache.k.shape)}")
for kv_cache_entry in kv_cache:
    print(f"Shape of cache entry for one layer = {tuple(kv_cache_entry.shape)}")
    break
print(f"Batch size = {kv_cache.batch}")
print(f"Current sequence length = {kv_cache.seq_len}")



























# Define new model parts where necessary, and create a new model & test it
# Note that sometimes our modules return a tuple of (tensor output, cache) rather than just output. The
# tests have been built to accommodate this.


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self,
        tokens: Int[Tensor, "batch position"],
        past_kv_pos_offset: int = 0
    ) -> Float[Tensor, "batch position d_model"]:

        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[past_kv_pos_offset: seq_len+past_kv_pos_offset],
            "seq d_model -> batch seq d_model",
            batch=batch
        )


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self,
        normalized_resid_pre: Float[Tensor, "batch posn d_model"],
        kv_cache_entry: KeyValueCacheTensor | None = None,
    ) -> tuple[
        Float[Tensor, "batch posn d_model"],
        KeyValueCacheTensor | None
    ]:
        '''
        Returns the result of applying attention layer to normlized_resid_pre, as well as
        the new cached key and value vectors (which we get from concatenating the old cached
        ones with the new key and value vectors).
        '''
        # Calculate the new query, key and value vectors
        q = einops.einsum(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
        ) + self.b_Q
        k = einops.einsum(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
        ) + self.b_K
        v = einops.einsum(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
        ) + self.b_V

        # If cache_entry is not None, this means we use the previous key and value vectors
        # Also we'll need to get a new cache entry which will be used later to construct a new cache
        if kv_cache_entry is not None:
            k = t.concat([kv_cache_entry[0], k], dim=1)
            v = t.concat([kv_cache_entry[1], v], dim=1)
            kv_cache_entry = t.stack([k, v])

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q, k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K"
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v, attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head"
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        out = einops.einsum(
            z, self.W_O,
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model"
        ) + self.b_O

        return out, kv_cache_entry

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Here, attn_scores have shape (batch, n_heads, query_pos, key_pos), where query_pos represents the
        new (non-cached) positions, and key_pos represent all the positions (cached and non-cached).

        So when we create our mask, the query indices and key indices will both go up to the same value
        (the full sequence length), but the query indices will start at >0.
        '''
        new_seq_len, full_seq_len = attn_scores.shape[-2:]
        assert new_seq_len <= full_seq_len
        q_posn = einops.repeat(attn_scores.new_tensor(range(full_seq_len-new_seq_len, full_seq_len)), "q -> q k", k=full_seq_len)
        k_posn = einops.repeat(attn_scores.new_tensor(range(full_seq_len)), "k -> q k", q=new_seq_len)
        mask = q_posn < k_posn
        attn_scores = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch position d_model"],
        kv_cache_entry: KeyValueCacheTensor | None = None,
    ) -> Float[Tensor, "batch position d_model"]:

        attn_out, kv_cache_entry = self.attn(self.ln1(resid_pre), kv_cache_entry)
        resid_mid = attn_out + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post, kv_cache_entry


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self,
        tokens: Int[Tensor, "batch seq_pos"],
        kv_cache: KeyValueCache | None = None
    ) -> Float[Tensor, "batch position d_vocab"]:

        using_kv_cache = kv_cache is not None

        if using_kv_cache:
            # If using kv_cache, then we only need to pass forward the newest tokens
            # Remember to add positional offset!
            n_cached_tokens = kv_cache.seq_len
            tokens = tokens[:, n_cached_tokens:]
            residual = self.embed(tokens) + self.pos_embed(tokens, n_cached_tokens)
        else:
            # If not using cache, turn it into a list of None's (so we can iterate through it)
            kv_cache = [None for _ in range(self.cfg.n_layers)]
            residual = self.embed(tokens) + self.pos_embed(tokens)

        # Apply all layers, and create a (new) kv_cache from the key & value vectors
        new_kv_cache_entries: list[KeyValueCacheTensor] = []
        for block, kv_cache_entry in zip(self.blocks, kv_cache):
            residual, kv_cache_entry = block(residual, kv_cache_entry)
            if using_kv_cache: new_kv_cache_entries.append(kv_cache_entry)

        logits = self.unembed(self.ln_final(residual))

        if using_kv_cache:
            return logits, KeyValueCache(t.stack(new_kv_cache_entries))
        else:
            return logits, None


tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

@t.inference_mode()
def sample_with_cache(
    self: TransformerSampler,
    prompt: str,
    max_tokens_generated=100,
    kv_cache: KeyValueCache | None = None,
    verbose=False,
    seed: int | None = None,
    **kwargs
) -> str:

    self.model.eval()
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    if seed is not None:
        np.random.seed(seed)
        t.manual_seed(seed)

    for i in tqdm(range(max_tokens_generated)):
        # Get new logits (make sure we don't pass in more tokens than the model's context length)
        logits, kv_cache = self.model(input_ids[None, -self.cfg.n_ctx:], kv_cache)
        # We only take logits for the last token, because this is what we're sampling
        logits = logits[0, -1]
        # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
        next_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
        # Create new input ids string, with shape (1, old_seq_len + 1)
        input_ids = t.cat([input_ids, next_token], dim=-1)
        # Print out results, if required
        if verbose:
            print(self.tokenizer.decode(input_ids), end="\r")
        # If our new token was the end-of-text token, stop
        if next_token == getattr(self.tokenizer, "eos_token_id", None):
            break

    return self.tokenizer.decode(input_ids)


TransformerSampler.sample = sample_with_cache

device = t.device("cuda") # can also try "cpu"

model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False);

initial_text = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
# input_ids = tokenizer.encode(initial_text, return_tensors="pt").squeeze()

sampler = TransformerSampler(model, tokenizer)

# Run the noncached version
t0 = time.time()
text = sampler.sample(
    initial_text,
    temperature=0.7,
    top_p=0.95,
    seed=0,
)
print(f"Time taken (without cache): {time.time() - t0:.2f} seconds")
rprint(f"Model output:\n\n[bold dark_orange]{text}[/]")

# Run the cached version
t0 = time.time()
text_with_cache = sampler.sample(
    initial_text,
    temperature=0.7,
    top_p=0.95,
    seed=0,
    kv_cache=KeyValueCache.new_empty(sampler.cfg)
)
print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
rprint(f"Model output:\n\n[bold dark_orange]{text_with_cache}[/]")

# # Check they are the same
assert text == text_with_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation (or failed to use random seeds)."
print("Tests passed!")




#Bonus part
@dataclass
class Beams:
    '''Class to store beams during beam search.'''
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]
    kv_cache: KeyValueCache | None = None

    def __getitem__(self, idx) -> "Beams":
        '''Helpful function allowing you to take a slice of the beams object along the batch dimension.'''
        return Beams(
            self.model,
            self.tokenizer,
            self.logprob_sums[idx],
            self.tokens[idx],
            self.kv_cache[:, :, idx] if self.kv_cache is not None else None
        )

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        '''Returns self as a list of logprob sums and completions (useful for getting final output).'''
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]


    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        '''
        Starting from the current set of beams (i.e. self.tokens) and returns a new set of `len(self.tokens) * k` beams,
        containing the best `k` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with a repeating n-gram
        of this length. 
        '''
        # Get the output logprobs for the next token (for every sequence in current beams)
        logprobs, kv_cache = self.model(self.tokens, self.kv_cache)
        logprobs = logprobs[:, -1, :].log_softmax(-1)

        # Get the top `toks_per_beam` tokens for each sequence
        topk_logprobs, topk_tokenIDs = self.get_topk_non_repeating(logprobs, no_repeat_ngram_size, k=k)

        # Add new logprobs & concat new tokens. When doing this, we need to add an extra `k` dimension since our current
        # logprobs & tokens have shape (batch,) and (batch, seq), but our new ones both have shape (batch, k)
        new_logprob_sums = einops.repeat(self.logprob_sums, "b -> b k", k=k) + topk_logprobs
        new_tokens = t.concat([einops.repeat(self.tokens, "b s -> b k s", k=k), topk_tokenIDs.unsqueeze(-1)], dim=-1)

        return Beams(self.model, self.tokenizer, new_logprob_sums.flatten(), new_tokens.flatten(0, 1), new_kv_cache)


    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        '''
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `k` which are also not terminated.
            early_terminations: Beams
                filtered version of self, containing all best `k` which are also terminated.
        '''
        # Get the indices of top `k` beams
        top_beam_indices = self.logprob_sums.topk(k=k, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = t.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # Get the indices of the `k` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # Return the beam objects from these indices
        return self[best_continuing], self[best_terminated]


    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: int | None,
        k: int,
    ) -> tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure that no returned tokens would produce an
            ngram of size  `no_repeat_ngram_size` which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape

        # If completion isn't long enough for a repetition, or we have no restrictions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate, by checking all past ngrams for a match
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1)  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated, -1.0e4, logprobs[range(batch), ngram_end_tokens]
                )

        return logprobs.topk(k=k, dim=-1)

    def print(self, title="Best completions", max_print_chars=80) -> None:
        '''
        Prints out a set of sequences with their corresponding logitsums.
        '''
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
        kv_cache: KeyValueCache | None = None,
    ) -> list[tuple[float, Tensor]]:
        '''
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        '''
        assert num_return_sequences <= num_beams
        self.model.eval()

        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        final_logprobs_and_completions = []  # we add to this list as we get terminated beams
        best_beams = Beams(self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens)  # start with just 1 beam

        for _ in tqdm(range(max_new_tokens)):
            # Generate & filter beams
            best_beams = best_beams.generate(k=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
            best_beams, best_beams_terminated = best_beams.filter(k=num_beams)

            # Add terminated beams to our list, and return early if we have enough
            final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)
            if len(final_logprobs_and_completions) >= num_return_sequences:
                return final_logprobs_and_completions[:num_return_sequences]

        # Return terminated beams plus the best ongoing beams of length `orig_len + max_new_tokens`
        final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
        return final_logprobs_and_completions[:num_return_sequences]






prompt = "For you, the day Bison graced your village was the most important day of your life. But for me, it was"
orig_len = len(tokenizer.encode(prompt))

beam_search_kwargs = dict(
    prompt=prompt,
    num_return_sequences=3,
    num_beams=20,
    max_new_tokens=60,
    no_repeat_ngram_size=2,
    verbose=False
)

sampler = TransformerSampler(model, tokenizer)

# Run the noncached version
t0 = time.time()
final_logitsums_and_completions = sampler.beam_search(**beam_search_kwargs)
logprob_sum, text = final_logitsums_and_completions[0]
avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp().item()
print(f"Time (without cache): {time.time() - t0:.2f} seconds")
print(f"Avg logprob (expressed as a probability) = {avg_logprob_as_prob:.3f}")
rprint(f"Output:\n\n[bold dark_orange]{text}[/]\n\n")

# Run the cached version
t0 = time.time()
beam_search_kwargs["kv_cache"] = KeyValueCache.new_empty(model.cfg)
final_logitsums_and_completions = sampler.beam_search(**beam_search_kwargs)
logprob_sum, text_with_cache = final_logitsums_and_completions[0]
avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp().item()
print(f"Time (with cache): {time.time() - t0:.2f} seconds")
print(f"Avg logprob (as probability) = {avg_logprob_as_prob:.3f}", end="")
rprint(f"Output:\n\n[bold dark_orange]{text_with_cache}[/]\n\n")

# Check they are the same
assert text == text_with_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation."
print("Tests passed!")