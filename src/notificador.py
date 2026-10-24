"""Envia avisos pelo Telegram (só texto: nenhuma imagem sai da casa).

Configuração no .env:
    TELEGRAM_TOKEN=...      (dado pelo @BotFather)
    TELEGRAM_CHAT_ID=...    (descubra rodando este arquivo, ver abaixo)

Teste:
    python src/notificador.py            # sem CHAT_ID: mostra os chats do bot
    python src/notificador.py "teste"    # com CHAT_ID: envia a mensagem
"""
import os
import sys
import threading

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"


def enviar(texto):
    """Envia e devolve True/False. Nunca levanta erro: uma falha de rede
    não pode derrubar o detector."""
    if not TOKEN or not CHAT_ID:
        print("Telegram não configurado no .env; aviso não enviado:", texto)
        return False
    try:
        r = requests.post(f"{API}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": texto}, timeout=10)
        if r.ok:
            return True
        print(f"Telegram recusou ({r.status_code}): {r.text[:200]}")
    except requests.RequestException as erro:
        print(f"Falha ao enviar pelo Telegram: {erro}")
    return False


def enviar_em_segundo_plano(texto):
    """Envia sem travar o vídeo: a requisição roda numa thread separada."""
    threading.Thread(target=enviar, args=(texto,), daemon=True).start()


def mostrar_chats():
    """Lista os chats que mandaram mensagem ao bot, para achar o CHAT_ID."""
    r = requests.get(f"{API}/getUpdates", timeout=10)
    if not r.ok:
        raise SystemExit(f"Token inválido? Resposta: {r.status_code}")
    chats = {}
    for u in r.json()["result"]:
        msg = u.get("message") or u.get("my_chat_member") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("title") or chat.get("first_name")
    if not chats:
        print("Nenhum chat ainda. Mande uma mensagem ao bot (ou adicione-o "
              "a um grupo e mande algo lá) e rode de novo.")
    for chat_id, nome in chats.items():
        print(f"TELEGRAM_CHAT_ID={chat_id}   ({nome})")


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Coloque TELEGRAM_TOKEN no .env primeiro.")
    if not CHAT_ID:
        mostrar_chats()
    else:
        texto = sys.argv[1] if len(sys.argv) > 1 else "Teste do fall_detection"
        print("Enviado!" if enviar(texto) else "Não enviado.")
