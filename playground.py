from base_model import BaseModel

if __name__ == "__main__":
    generator = BaseModel()

    answer, system_prompt, user_prompt = generator.generate_answer(
        user_question="Quel est le plus beau bâtiment de Paris ?"
    )

    print(answer)
