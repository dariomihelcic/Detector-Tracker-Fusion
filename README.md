# Detector-Tracker-Fusion
Fusion and evaluation of a custom detector and several tracking methods.

A new dataset with seven classes was created. Data for two classes were manually collected and annotated, while the data for other classes were taken from public repositories.
The use of a single detector as a tracking method was compared to three other common trackers: KCF, CRST and MedianFlow.

The best result (IoU score) was achieved with a combination of methods, where the tracker was periodically updated with detection results.

<img src="https://github.com/dariomihelcic/Detector-Tracker-Fusion/blob/main/doc/det_tracking_figure.PNG" />
