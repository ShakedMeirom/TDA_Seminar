if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    import time
    from Original_Accuracy_test import accuracy_test
    from high_persistence_test import high_persistence_test

    # ensure chofer_torchex and pershombox are available
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.path.join(os.path.dirname(__file__)))

    sys.path.append(os.path.join(parent, 'chofer_torchex'))
    sys.path.append(os.path.join(parent, 'tda-toolkit'))

    from src.reddit_12K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_12K.experiments import experiment

    from src.sharedCode.data_downloader import download_provider, download_raw_data
    from src.sharedCode.gui import ask_user_for_provider_or_data_set_download

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/reddit_12K.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/reddit_12K/reddit_subreddit_10K.graph')

    if not os.path.isfile(provider_path):

        choice = ask_user_for_provider_or_data_set_download()

        if choice == "download_data_set":
            download_raw_data("reddit_12K")
            generate_dgm_provider(raw_data_path,
                                  provider_path)

        elif choice == "download_provider":
            download_provider('reddit_12K')
            time.sleep(1)  # included since sometimes downloaded file is not yet available when experiment starts.

    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')

    high_persistence_test(experiment, provider_path)
    accuracy_test(experiment, provider_path)

