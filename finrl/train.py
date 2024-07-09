from __future__ import annotations

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config import DATA_SOURCE

# construct environment
async def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)

    data = await dp.download_data(ticker_list=ticker_list, start_date=start_date, end_date=end_date, time_interval=time_interval)
    print(f'Downloading data for ticker_list {ticker_list}, start_date {start_date}, end_date {end_date}, and time_interval {time_interval}')
    print("Raw data columns:", data.columns)
    print("Raw data head:")
    print(data.head())
    
    data = dp.clean_data(data)
    print("Cleaned data columns:", data.columns)
    print("Cleaned data head:")
    print(data.head())
    
    data = dp.add_technical_indicator(data, tech_indicator_list=technical_indicator_list)
    print("Data with indicators columns:", data.columns)
    print("Data with indicators head:")
    print(data.head())
    
    if if_vix:
        data = dp.add_vix(data)
        print("Final DataFrame columns:", data.columns)
        print("Final DataFrame head:")
        print(data.head())
    else:
        data = dp.add_turbulence(data)
        print("Final DataFrame columns:", data.columns)
        print("Final DataFrame head:")
        print(data.head())
    
    try:
        price_array, tech_array, turbulence_array  = dp.df_to_array(data, if_vix)
        print("Price array shape:", price_array.shape)
        print("Tech array shape:", tech_array.shape)
        print("Turbulence array shape:", turbulence_array.shape)
 
        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": True,
        }
    except ValueError as e:
        if "not enough values to unpack" in str(e):
            price_array, tech_array = dp.df_to_array(data, if_vix=False)
            print("Price array shape:", price_array.shape)
            print("Tech array shape:", tech_array.shape)
 
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": None,
                "if_train": True,
            }
        else:
            raise e

    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model, model_config = agent_rllib.get_model(model_name)
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #
    # # demo for stable-baselines3
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac",
    #     agent_params=SAC_PARAMS,
    #     total_timesteps=1e4,
    # )
