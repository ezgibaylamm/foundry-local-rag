from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="foundry-local-rag")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("Foundry Local başarıyla başlatıldı.")
    print(manager)


if __name__ == "__main__":
    main()