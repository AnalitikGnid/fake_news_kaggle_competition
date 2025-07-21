import pandas as pd
import tiktoken
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import numpy as np
import os

# Load data
f = pd.read_csv('final_cleaned_combined_data.csv')

# Check required columns
assert 'text' in f.columns and 'label' in f.columns, "Dataset must contain 'text' and 'label' columns"

# Tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

# Tokenize text column
f['tokens'] = f['text'].astype(str).apply(lambda x: enc.encode(x))
f['token_count'] = f['tokens'].apply(len)

# Split into real and fake
real = f[f['label'] == 1]
fake = f[f['label'] == 0]

### ----- 1. Plot and Save Token Count Distribution -----
plt.figure(figsize=(10, 6))
sns.histplot(real['token_count'], color='blue', label='Real', kde=True)
sns.histplot(fake['token_count'], color='red', label='Fake', kde=True)
plt.legend()
plt.title("Token Count Distribution: Real vs Fake News")
plt.xlabel("Number of Tokens")
plt.ylabel("Article Count")
plt.tight_layout()
plt.savefig("token_count_distribution.png")
plt.show()

### ----- 2. Frequency Analysis -----
# Flatten token lists
real_tokens = [t for tokens in real['tokens'] for t in tokens]
fake_tokens = [t for tokens in fake['tokens'] for t in tokens]

real_freq = Counter(real_tokens)
fake_freq = Counter(fake_tokens)

# Merge into one DataFrame
all_tokens = set(real_freq) | set(fake_freq)
data = []
for t in all_tokens:
    r = real_freq.get(t, 0)
    f = fake_freq.get(t, 0)
    data.append((t, r, f, abs(r - f)))

df = pd.DataFrame(data, columns=['token', 'real_freq', 'fake_freq', 'freq_diff'])
df['token_str'] = df['token'].apply(lambda t: enc.decode([t]))
df = df.sort_values('freq_diff', ascending=False)

top_diff = df.head(20)

print("\n Top 20 Differentiating Tokens:")
print(top_diff[['token_str', 'real_freq', 'fake_freq', 'freq_diff']].to_string(index=False))

### ----- 3. Bar Plot: Top 20 Differentiating Tokens -----
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_diff,
    x='token_str', y='freq_diff',
    palette='magma'
)
plt.xticks(rotation=45)
plt.title("Top 20 Differentiating Tokens by Frequency Difference")
plt.xlabel("Token")
plt.ylabel("Absolute Frequency Difference")
plt.tight_layout()
plt.savefig("top_token_differences.png")
plt.show()

### ----- 4. KL Divergence -----
real_vec = np.array([real_freq.get(t, 0) + 1 for t in all_tokens])  # +1 smoothing
fake_vec = np.array([fake_freq.get(t, 0) + 1 for t in all_tokens])

real_prob = real_vec / real_vec.sum()
fake_prob = fake_vec / fake_vec.sum()

kl_real_fake = entropy(real_prob, fake_prob)
kl_fake_real = entropy(fake_prob, real_prob)

print(f"KL(real || fake): {kl_real_fake:.4f}")
print(f"KL(fake || real): {kl_fake_real:.4f}")

### ----- 5. Save CSV Output -----
df_sorted = df.sort_values('freq_diff', ascending=False)
df_sorted.to_csv("token_frequency_comparison.csv", index=False)

