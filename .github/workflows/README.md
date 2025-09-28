# My Lichess Chess Bot ♟️

This is a custom chess bot that runs on Lichess using [python-chess](https://python-chess.readthedocs.io/) and the [Lichess Bot API](https://lichess.org/api#tag/Bot).

## Features
- Dynamic search depth (starts at 10 ply and grows as pieces are traded)
- Evaluation based on:
  - Pawn structure (doubled, isolated, backward pawns)
  - Square control with pawn diagonal rules
  - Pawn advancement bonuses
  - King safety (control of surrounding squares)
- Alpha-beta pruning that activates at deeper depths
- Logs every considered move with evaluation
- Accepts *any* challenge on Lichess automatically

## Setup

1. **Create a bot account on Lichess**  
   - Register a new Lichess account for your bot.  
   - Apply for [Bot status](https://lichess.org/account/oauth/bot).  
   - Generate a personal access token with scope `bot:play`.

2. **Fork or clone this repo**  

3. **Add your Lichess token to GitHub**  
   - Go to your repo → Settings → Secrets and variables → Actions.  
   - Add a new secret named **`LICHESS_TOKEN`** and paste your token value.

4. **Push to GitHub**  
   - Commit `bot.py`, `requirements.txt`, and `.github/workflows/bot.yml`.  
   - GitHub Actions will start the bot automatically.

5. **Play the bot**  
   - Once Actions is running, go to your bot’s Lichess profile.  
   - Send it a challenge — it will accept and play automatically.

## Local Testing
If you want to test locally:
```bash
pip install -r requirements.txt
export LICHESS_TOKEN=your_token_here
python bot.py
