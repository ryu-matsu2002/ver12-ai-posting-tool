import asyncio
from app.services.agent.livedoor_gpt_agent import LivedoorAgent
from app.services.mail_utils.mail_gw import create_inbox

async def main():
    site = "test-site"
    nickname = "testuser" + str(__import__("random").randint(1000, 9999))
    password = "safe" + nickname  # livedoor ID含まないよう注意
    email, token = create_inbox()

    print(f"[TEST] email: {email}")
    print(f"[TEST] nickname: {nickname}")
    print(f"[TEST] password: {password}")

    agent = LivedoorAgent(site, email, password, nickname, token)
    result = await agent.run()
    print(f"[RESULT] {result}")

if __name__ == "__main__":
    asyncio.run(main())
