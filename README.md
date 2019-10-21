Dataset Generator utility for the ADFA Torch project.

# ADFA Torch Set

ADFA Torch set, for Audio-Driven Facial Animation Pytorch Dataset, is a simple utility for generating the dataset of the ADFA project.
It uses the RAVDESS Dataset [[1]] and the 3DDFA Facial Reconstructor [[2]] to generate the appropriate data composed of *.wav* audio files and *.npz* facial vertices for each videos.

## Install

The install consists of a simple git clone and a pip3 command in order to install all dependencies ( may require to run as sudo )

```bash
cd ./ADMATORCHSET
pip3 install -r requirements.txt
```

## Usage

```bash
python3 -m adfatorchset.main \
  -i input_dir \  # Input Directory to save the RAVDESS files
  -o output_dir \ # Output Director to save the extracted data
  -f 30 \         # Frame Rate for the frames extraction
  -s 16000 \      # Sample Rate for the audio extraction
  -b 32 \         # Batch size for facial reconstruction inference
  -c \            # Cuda enable flag
  -v \            # Verbose enable for ffmpeg
  -r              # Download RAVDESS flag
```

## Example

Example of generated output ( rendered beside original video ) applied on custom data ( not RAVDESS for testing purposes ).

![ Demo ]( ./demo.gif )

## References

* [[1]] The PyTorch improved version of TPAMI 2017 paper: Face Alignment in Full Pose Range: A 3D Total Solution
* [[2]] The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English

[1]: https://github.com/cleardusk/3DDFA
[2]: https://doi.org/10.5281/zenodo.1188976
