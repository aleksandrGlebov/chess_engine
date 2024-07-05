import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle

# Модель нейронной сети
class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class ChessModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=3):
        super(ChessModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(2000, embedding_dim)  # Максимальная длина партии
        self.elo_embedding = nn.Embedding(3000, embedding_dim)  # Максимальный рейтинг
        
        self.lstm = nn.LSTM(embedding_dim * 3, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention_modules = nn.ModuleList([AttentionModule(hidden_dim * 2) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, moves, white_elo, black_elo):
        batch_size, seq_len = moves.size()
        
        move_emb = self.embedding(moves)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=moves.device).unsqueeze(0).repeat(batch_size, 1))
        white_elo_emb = self.elo_embedding(white_elo).unsqueeze(1).repeat(1, seq_len, 1)
        black_elo_emb = self.elo_embedding(black_elo).unsqueeze(1).repeat(1, seq_len, 1)
        
        x = torch.cat([move_emb, pos_emb, (white_elo_emb + black_elo_emb) / 2], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        
        for attention_module in self.attention_modules:
            lstm_out = attention_module(lstm_out)
        
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

# Функция для обработки PGN файла
def process_game(game_text):
    try:
        lines = game_text.strip().split('\n')
        metadata = {}
        moves = []
        
        # Extract metadata
        for line in lines:
            if line.startswith('[') and ']' in line:
                key_value = line[1:-1].split('"')
                key = key_value[0].strip()
                value = key_value[1].strip() if len(key_value) > 1 else ""
                metadata[key] = value
            elif not line.startswith('['):
                # This line contains moves
                move_text = re.sub(r'\{[^}]*\}', '', line)  # Remove comments
                move_text = re.sub(r'\d+\.\.\.', '', move_text)  # Remove move numbers for black
                move_text = re.sub(r'\d+\.', '', move_text)  # Remove move numbers
                moves.extend(move_text.split())

        if not moves:
            print(f"No moves found for game: {metadata.get('Event', 'Unknown event')}")
            print(f"Game text: {game_text[:200]}...")  # Print first 200 characters of the game text
            return None

        return {
            "metadata": metadata,
            "moves": moves,
        }
    except Exception as e:
        print(f"Error processing game: {str(e)}")
        print(f"Game text: {game_text[:200]}...")  # Print first 200 characters of the game text
        return None

# Параллельная обработка PGN файла
def parallel_process_pgn(file_path, max_games=None):
    processed_games = []
    with open(file_path, 'r') as pgn_file:
        current_game = []
        games_processed = 0

        for line in tqdm(pgn_file, desc="Reading PGN file"):
            if line.strip() == "" and current_game:
                game_text = '\n'.join(current_game)
                processed_game = process_game(game_text)
                if processed_game:
                    processed_games.append(processed_game)
                    games_processed += 1
                    if max_games and games_processed >= max_games:
                        break
                current_game = []
            else:
                current_game.append(line.strip())

        if current_game:
            game_text = '\n'.join(current_game)
            processed_game = process_game(game_text)
            if processed_game:
                processed_games.append(processed_game)

    return processed_games

# Класс для работы с данными
class ChessDataset(Dataset):
    def __init__(self, data, max_seq_length=100):
        self.data = data
        self.move_to_index = self._create_move_index()
        self.max_seq_length = max_seq_length

    def _create_move_index(self):
        all_moves = set()
        for game in self.data:
            all_moves.update(game['moves'])
        return {move: idx + 1 for idx, move in enumerate(sorted(all_moves))}  # Start from 1, reserve 0 for padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game = self.data[idx]
        moves = [self.move_to_index.get(move, 0) for move in game['moves']]
        
        # Pad or truncate the moves sequence
        if len(moves) > self.max_seq_length:
            moves = moves[:self.max_seq_length]
        else:
            moves = moves + [0] * (self.max_seq_length - len(moves))
        
        white_elo = int(game['metadata'].get('WhiteElo', '0')) if game['metadata'].get('WhiteElo', '0').isdigit() else 0
        black_elo = int(game['metadata'].get('BlackElo', '0')) if game['metadata'].get('BlackElo', '0').isdigit() else 0
        
        return {
            'moves': torch.tensor(moves[:-1], dtype=torch.long),
            'target': torch.tensor(moves[1:], dtype=torch.long),
            'white_elo': torch.tensor(white_elo, dtype=torch.long),
            'black_elo': torch.tensor(black_elo, dtype=torch.long),
        }

# Функция для обучения модели
def train_model(model, dataloader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                moves = batch['moves'].to(device)
                target = batch['target'].to(device)
                white_elo = batch['white_elo'].to(device)
                black_elo = batch['black_elo'].to(device)
                
                optimizer.zero_grad()
                output = model(moves, white_elo, black_elo)
                loss = criterion(output.transpose(1, 2), target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        scheduler.step(avg_loss)

# Основная функция
def main():
    pgn_file_path = 'E:/sky/lichess_db_standard_rated_2018-06.pgn'
    processed_data_file = 'processed_games_10000.pkl'
    model_save_path = 'chess_model_10000.pth'
    max_games = 10000
    max_seq_length = 100  # Или любое другое подходящее значение

    if not os.path.exists(processed_data_file):
        print("Processing PGN file...")
        processed_games = parallel_process_pgn(pgn_file_path, max_games=max_games)
        
        if len(processed_games) == 0:
            print("No valid games were processed.")
            return

        with open(processed_data_file, 'wb') as f:
            pickle.dump(processed_games, f)
        print(f"Processed data saved to {processed_data_file}")
    else:
        print("Loading preprocessed data...")
        with open(processed_data_file, 'rb') as f:
            processed_games = pickle.load(f)
    
    if len(processed_games) == 0:
        print("No valid games found in the preprocessed data.")
        return
    
    print(f"Number of processed games: {len(processed_games)}")
    
    # Загрузка данных
    dataset = ChessDataset(processed_games, max_seq_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Инициализация модели
    vocab_size = len(dataset.move_to_index) + 1  # +1 for padding
    model = ChessModel(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512)
    
    # Обучение модели
    print("Training model...")
    train_model(model, dataloader, num_epochs=50, learning_rate=0.001)
    
    # Сохранение модели
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()