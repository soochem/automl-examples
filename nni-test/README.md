# nni-test

* Target
    - NNI 1.4
* Function
    - AutoFE : Automated Feature Engineering
        - Feature Selection
        - Feature Extraction
    - HPO : Hyperparameter Optimization
        - Exhaustive Search
        - Heuristic Search
        - Bayesian Optimization
        - Reinforcement Learning
    - NAS : Neural Architecture Search
    - Model Compression


* Error Report
    - Error while installing nni
        - sudo yum install gcc 
            - if your python doesn't have "python header file (Python.h)", sudo yum install python3-devel

    - Error while constructing remote training service
        - Message : "error":"TrainingService setClusterMetadata timeout. Please check your config file."
        - In my case, I was working on Google Cloud Platform (GCE VM) and the config about RSA auth was wrong.
        - You can find various ways of authorization here : https://github.com/microsoft/nni/blob/master/docs/en_US/Tutorial/ExperimentConfig.md#sshkeypath
        - Solution : I added two lines, "sshKeyPath: ~/.ssh/id_rsa" and "passphrase: {your_pass}", to my config.yml file, and it worked fine.
             - If you're using OpenSSH, be careful about your SSH key name (e.g. id_rsa is fine)
        - In summary, it wasn't an actual error. It worked well with config.yml templates.
        
        - ps. If you want to use GCE to make nni job clusters, there are some steps to follow.
            1) Let's assume that we have 3 nodes. One of the nodes is "master" and the others are "workers"
            2) First, create SSH key (ssh-keygen) at "master" node and upload this public key(e.g. id_rsa.pub) to GS (or any place to store)
            3) Then, download this key to your worker nodes.
            4) Add this to ~/.ssh/authorized_keys.
            5) Try ssh -v user_name@ip to check if the key's working.