from argus.main import main, load_config

if __name__ == "__main__":
    # Load configuration from the file in the same directory
    config = load_config("configs/config.yml")
    main(config)
