

def load_and_align_minian(
    mouse, session, neural_type="spikes", sigma=None, sampling_rate=1 / 30
):
    """
    Parameters:
    ==========
    mouse : str
        name of the mouse (e.g. 'RLI5-1')
    session : str
        one of ['NeutralExposure', 'FC', 'Recall1', 'Recall2', 'Recall3', 'OfflineDay1', 'OfflineDay2']
    neural_type : str
        one of ['traces', 'spikes', 'smoothed']
    sigma : int
        smoothing kernel if neural_type=smoothed
    """

    # create time vector based on expected length of session
    if "NeutralExposure" in session:
        frame_count = (
            10 * 60 / sampling_rate
        )  # 10min x 60sec/min / sampling_rate (usually 1/30)
    elif "Offline" in session:
        frame_count = 60 * 60 / sampling_rate
    elif "Recall" in session:
        frame_count = 5 * 60 / sampling_rate
    elif "FC" in session:
        frame_count = 274 / sampling_rate
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['NeutralExposure', 'FC', 'Recall1', 'Recall2', 'Recall3', 'OfflineDay1', 'OfflineDay2']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)

    # load the specified type of neural activity
    dpath = get_dpath(mouse, session)
    mouse_minian = open_minian(dpath + "/minian", return_dict=True)
    if neural_type == "traces":
        neural_activity = mouse_minian["C"]
    elif (neural_type == "spikes") or (neural_type == "smoothed"):
        neural_activity = mouse_minian["S"]
    else:
        raise Exception(
            "Not a valid 'neural_type'; must be one of ['traces', 'spikes', 'smoothed']."
        )

    minian_timestamps = pd.read_csv(dpath + "/timeStamps.csv")
    lined_up_timeframes = align_miniscope_frames(minian_timestamps, time)
    neural_activity = neural_activity.sel(frame=lined_up_timeframes)
    if (
        neural_type == "smoothed"
    ):  # this filtering must be done after the previous line because this converts neural_activity to numpy array
        neural_activity = gaussian_filter(neural_activity, sigma=(1, sigma))

    return neural_activity