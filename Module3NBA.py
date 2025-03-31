import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


#Read and clean dataset
df = pd.read_csv('player_data.csv')
df['Colleges'] = df['Colleges'].fillna("none")
df['Ht'] = df['Ht'].apply(lambda x: int(x.split('-')[0])*12 + int(x.split('-')[1]))
df['Pos'] = df['Pos'].apply(lambda x: list(x.replace('-', '')))
df = df.dropna()

#Prepare and normalize stats
stats = df[['From', 'To', 'Ht', 'Wt']]
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(stats.iloc[:, 1:])
stats_normalized = pd.DataFrame(stats_scaled, index=df['Player'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(stats_normalized)

# Get player name input from the user
player_input = input("Enter the player name to find the most similar players: ")

# Check if the player exists in the dataframe
if player_input not in df['Player'].values:
    print(f"{player_input} is not in the player list. Please check the spelling.")
else:
    player_idx = df[df['Player'] == player_input].index[0]
    similarity_scores = list(enumerate(cosine_sim[player_idx]))

    # Sort the players by similarity score (excluding the player itself)
    sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]
    similar_players = [df.iloc[i[0]]['Player'] for i in sorted_similar]

    print(f"\nTop 10 similar players to {player_input}:")
    for i in sorted_similar:
        player_idx = i[0]
        similarity_score = i[1]
        similar_player_name = df.iloc[player_idx]['Player']
        print(f"{similar_player_name}: Similarity Score = {similarity_score:.6f}")
