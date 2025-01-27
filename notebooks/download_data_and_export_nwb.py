




from pathlib import Path
from datetime import timedelta
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals







GENOTYPES = ["wt/wt"]
SESSION_TYPES = ["brain_observatory_1.1"]
BRAIN_AREAS = ["VISpm"]
STIMULUS_TYPES = ["gabors", "flashes"]
CLEAR_CACHE = True







#| eval: false
root = Path(__file__).parent.parent.absolute()



cache_dir = root/"data"/"cache"
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)






cache = EcephysProjectCache.from_warehouse(manifest=cache_dir/"manifest.json")
sessions = cache.get_session_table()
sessions = sessions[sessions.full_genotype.isin(GENOTYPES)]
sessions = sessions[sessions.session_type.isin(SESSION_TYPES)]
print(f"Selected {len(sessions)} sessions")
# sessions = sessions.iloc[:1] # for testing purposes, just do a single session







channels = cache.get_channels()
channels = channels[channels.ecephys_structure_acronym.isin(BRAIN_AREAS)]
channels = channels[channels.ecephys_session_id.isin(sessions.index)]







#| label: ses-loop
for i_ses, session_id in enumerate(sessions.index):
    print(f"##### Sub {str(i_ses+1).zfill(2)} #####")
    session = cache.get_session_data(session_id)
    metadata = session.metadata








#| ref.label: ses-loop
    subject = Subject(
        subject_id = str(i_ses+1).zfill(2),
        age=f"P{int(metadata['age_in_days'])}",
        species = "Mus musculus",
        sex = metadata['sex'],
        genotype = metadata['full_genotype'],
        description = f"mouse {metadata['specimen_name']}"
)







#| ref.label: ses-loop
    ses_channels = session.channels
    ses_channels = ses_channels[ses_channels.ecephys_structure_acronym.isin(BRAIN_AREAS)]
    probes = ses_channels.probe_id.unique()








#| ref.label: ses-loop
    #| label: run-loop
    epochs = session.get_stimulus_epochs()
    epochs = epochs[epochs.stimulus_name.isin(STIMULUS_TYPES)]
    for i_epoch in range(len(epochs)):
        print(f"##### Run {i_epoch+1} #####")
        epoch = epochs.iloc[i_epoch]
        t_start = metadata["session_start_time"]+timedelta(seconds=epoch.start_time)









#| ref.label: ses-loop
    #| ref.label: run-loop
        nwb = NWBFile(
            session_description="Ecephys recordings for multiple Neuropixels probes",
            identifier=f"session{session_id}_run{i_epoch+1}",
            session_start_time=metadata["session_start_time"],
        )
        nwb.subject = subject
        ecephys_module = nwb.create_processing_module(
            name="ecephys", description="processed extracellular electrophysiology data"
        )
        out_fname = f"sub-{subject.subject_id}_run-{i_epoch+1}_ecephys.nwb"
        out_fpath = root/"data"/"nwb"/f"sub-{subject.subject_id}"/out_fname
        if not out_fpath.parent.exists():
            out_fpath.parent.mkdir(parents=True)
        







#| ref.label: ses-loop
    #| ref.label: run-loop
        #| label: probe-loop
        for i_probe, probe_id in enumerate(probes):
            lfp = session.get_lfp(probe_id)
            probe_channels = ses_channels[ses_channels.probe_id==probe_id]
            valid_channel_ids = []
            for pc_id in probe_channels.index.to_numpy():
                if np.isin(pc_id, lfp.channel):
                    valid_channel_ids.append(pc_id)
            probe_channels = probe_channels[probe_channels.index.isin(valid_channel_ids)] 
            times = lfp.time[np.logical_and(epoch.start_time<lfp.time, lfp.time<epoch.stop_time)]
            lfp = lfp.sel(channel=probe_channels.index, time=times)







#| ref.label: ses-loop
    #| ref.label: run-loop
        #| ref.label: probe-loop
            device = nwb.create_device(
                name=session.probes.loc[probe_id].description, description=f"ID {probe_id}", manufacturer="Neuropixels"
            )
            for i_loc, location in enumerate(probe_channels.ecephys_structure_acronym.unique()):
                electrode_group = nwb.create_electrode_group(
                    name=location,
                    description=f"Electrodes in {location}",
                    device=device, 
                    location=location
                )
                location_channels = probe_channels[probe_channels.ecephys_structure_acronym==location]
                for electrode_id in location_channels.index:
                    nwb.add_electrode(
                        group=electrode_group,
                        id = electrode_id,
                        location=location,
                    )







#| ref.label: ses-loop
    #| ref.label: run-loop
        #| ref.label: probe-loop
            region=list(np.where(nwb.electrodes.id[:] == location_channels.index)[0])
            electrode_table_region = nwb.create_electrode_table_region(
                description="all electrodes", region=region,
            )

            lfp = ElectricalSeries(
                name=f"lfp{session.probes.loc[probe_id].description[-1]}",
                description="LFP data",
                data=lfp.data,
                electrodes=electrode_table_region,
                starting_time=0.0,
                rate = 1/np.diff(lfp.time)[0]
            )
            ecephys_module.add(lfp);
        




# lfp = LFP(electrical_series=lfp)






#| ref.label: ses-loop
    #| ref.label: run-loop
        units = session.units
        units = units[units.ecephys_structure_acronym.isin(BRAIN_AREAS)]
        spike_times = session.spike_times
        spike_times = {uid: spike_times[uid] for uid in units.index}
        for uid, spikes in spike_times.items():
            spike_times[uid] = spikes[np.logical_and(
                    spike_times[uid]>=epoch.start_time,
                    spike_times[uid]<=epoch.stop_time,
                )]
            spike_times[uid]-=spike_times[uid][0]  # start at 0





#| ref.label: ses-loop
    #| ref.label: run-loop
        mean_waveforms = session.mean_waveforms
        mean_waveforms = {uid: mean_waveforms[uid] for uid in units.index}
        max_waveforms = {}
        for uid, mean_waveform in mean_waveforms.items():
            mean_waveform = mean_waveform.sel(channel_id=nwb.electrodes.id[:])
            peak_to_peak = mean_waveform.data.max(axis=1) - mean_waveform.data.min(axis=1)
            max_waveforms[uid] = mean_waveform.data[np.argmax(peak_to_peak)]






#| ref.label: ses-loop
    #| ref.label: run-loop
        for uid in units.index:
            nwb.add_unit(spike_times=spike_times[uid], waveform_mean=max_waveforms[uid], id=uid)







#| ref.label: ses-loop
    #| ref.label: run-loop
        running_speed = session.running_speed
        running_speed = running_speed[running_speed.start_time>=epoch.start_time]
        running_speed = running_speed[running_speed.end_time<=epoch.stop_time]
        running_intervals = TimeIntervals(
            name="running",
            description="Intervals when the animal was running.",
        )
        running_intervals.add_column(name="velocity", description="Running velocity in cm/s.")
        for i, rs_row in running_speed.iterrows():
            running_intervals.add_row(start_time=rs_row.start_time, stop_time=rs_row.end_time, velocity=rs_row.velocity)
        nwb.add_time_intervals(running_intervals)






#| ref.label: ses-loop
    #| ref.label: run-loop
        stimuli = session.get_stimulus_table()
        stimuli = stimuli[stimuli.start_time>=epoch.start_time]
        stimuli = stimuli[stimuli.stop_time<=epoch.stop_time]
        stim_type = stimuli.stimulus_name.unique()
        assert len(stim_type)==1
        stim_type = stim_type[0]







#| ref.label: ses-loop
    #| ref.label: run-loop
        if stim_type == "flashes":
            nwb.add_trial_column(name="color", description="Stimulus color (1=white, -1=black)")
            for _, stim in stimuli.iterrows():
                nwb.add_trial(start_time=stim.start_time, stop_time=stim.stop_time, color=stim.color)







#| ref.label: ses-loop
    #| ref.label: run-loop
        if stim_type == "gabors":
            nwb.add_trial_column(
                name="x_position",
                description="x coordiante of stimulus center is degrees of visual angle."
            )
            nwb.add_trial_column(
                name="y_position",
                description="y coordiante of stimulus center is degrees of visual angle."
            )
            nwb.add_trial_column(
                name="orientation",
                description="Orientation of the stimulus is degrees."
            )
            for _, stim in stimuli.iterrows():
                nwb.add_trial(
                    start_time=stim.start_time,
                    stop_time=stim.stop_time,
                    x_position=stim.x_position,
                    y_position=stim.y_position,
                    orientation=stim.orientation
                )





#| ref.label: ses-loop
    #| ref.label: run-loop
        with NWBHDF5IO(out_fpath, mode="w") as io:
            io.write(nwb)
