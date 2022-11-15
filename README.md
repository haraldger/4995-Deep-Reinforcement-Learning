## How To Use The Code

1. Clone the prooject
    ```
    git clone git@github.com:haraldger/4995-Deep-Reinforcement-Learning.git
    ```

2. Navigate into the project
    ```
    cd 4995-Deep-Reinforcement-Learning
    ```

3. Create a virtual environment
    ```
    python3 -m venv venv
    ```

4. Activate the virtual environment
    ```
    source venv/bin/activate
    ```

5. Upgrade pip
    ```
    pip install --upgrade pip
    ```

6. Install dependencies
    ```
    pip install -r requirements.txt
    ```
    
6. If you want to add a new dependency
    - First install it
        ```
        pip install <package_name>
        ```

    - Then, update the dependencies file
        ```
        pip freeze > requirements.txt
        ```

## Run The Mario Emulator
Run the following script to test the mario emulator

```
gym_super_mario_bros -e SuperMarioBrosRandomStages-v0 -m human --stages 1-1
```
Refer to https://github.com/Kautenja/gym-super-mario-bros for more information.


