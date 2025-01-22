import os
import sys
from glob import glob

include: "Snakefile_accessory_image_registration_06052022.py"

# -----------------------------------
# Parameters and executables:
# see Snakefile_accessory_image_registration.py

# -----------------------------------

rule all:
  input: expand(WORKING_DIR + '/image_processing/{sample}/{sample}_output_log.txt',sample=SAMPLES)
  params: name='all', partition='quake,owners', mem='1024'

rule pull:
  """ Get raw image stacks from the google drive
  """
  input: 'sample_list_processing_06052022.txt'
  output: directory(WORKING_DIR + '/image_processing_inputs/{sample}'), WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T5T6T7.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T1T2T3.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T9T10T11.czi'
  resources: mem_mb = '32000', disk_mb = '32000'
  params: name='pull', partition='quake,owners',mem='32000',time='12:00:00'
  threads: 1
  run:
    gdrive_filepath = GDRIVE_DIR + '/' + wildcards.sample
    dest_filepath = WORKING_DIR + '/image_processing_inputs/' + wildcards.sample
    file_prefix = ('_').join(wildcards.sample.split('_')[1:])
    dest1 = dest_filepath + '/' + file_prefix + '_T5T6T7-Airyscan Processing.czi'
    dest2 = dest_filepath + '/' + file_prefix + '_T1T2T3-Airyscan Processing.czi'
    dest3 = dest_filepath + '/' + file_prefix + '_T9T10T11-Airyscan Processing.czi'
    shell("module load system rclone && "
          "rclone copy gdrive_remote:'{gdrive_filepath}' {output[0]} && "
          "mv '{dest1}' {output[1]} && "
          "mv '{dest2}' {output[2]} && "
          "mv '{dest3}' {output[3]}")

rule register:
  """ Register the DAPI channels and save the displacements and rotations
  """
  input: WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T5T6T7.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T1T2T3.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T9T10T11.czi'
  output: WORKING_DIR + '/temp_files/{sample}/{sample}-overall_shifts.txt'
  resources: mem_mb = '32000'
  params: name='register', partition='quake,owners',mem='32000',time='04:00:00'
  threads: 1
  run:
    register_dapi_cluster_rotation.register_dapi(WORKING_DIR + '/temp_files', input[0],[input[1],input[2]],output[0],wildcards.sample)

rule save_registered_stacks:
  """ Use the calculated registration to align the channels, and save a new .tif stack for each channel
  """
  input: WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T5T6T7.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T1T2T3.czi',WORKING_DIR + '/image_processing_inputs/{sample}/{sample}_T9T10T11.czi',
         WORKING_DIR + '/temp_files/{sample}/{sample}-overall_shifts.txt',
  output: [WORKING_DIR + '/temp_files/{sample}/{sample}-channel-' + channel + '-registered.tif' for channel in ['0','1','2','3','4','5','6','7','8','9','10','11']],[WORKING_DIR + '/temp_files/{sample}/{sample}-channel-' + channel  + '-registered.-maxz.tif' for channel in ['0','1','2','3','4','5','6','7','8','9','10','11']]
  resources: mem_mb = '32000'
  params: name='save_registered', partition='quake,owners',mem='32000',time='08:00:00'
  threads: 1
  run:
      register_dapi_cluster_rotation.save_registered_stacks(working_dir=WORKING_DIR + '/temp_files',stack_list=input[:3],sample_handle=wildcards.sample,overall_shift_file=input[3])

rule copy_to_gdrive:
  """ Copy the registered .tif stacks to google drive. Output a log file as a marker that all steps have been completed.
  """
  input: [WORKING_DIR + '/temp_files/{sample}/{sample}-channel-' + channel + '-registered.tif' for channel in ['0','1','2','3','4','5','6','7','8','9','10','11']],
          [WORKING_DIR + '/temp_files/{sample}/{sample}-channel-' + channel  + '-registered.-maxz.tif' for channel in ['0','1','2','3','4','5','6','7','8','9','10','11']]
  output: WORKING_DIR + '/image_processing/{sample}/{sample}_output_log.txt'
  resources: mem_mb = '16000'
  params: name='push', partition='quake,owners',mem='16000',time='24:00:00'
  threads: 1
  run:
    oak_dirname = WORKING_DIR + '/temp_files/' + wildcards.sample
    shell("module load system rclone && "
          "rclone copy '{oak_dirname}' gdrive_remote:zfish_LPS_RNAscope/registered/{wildcards.sample}")
    write_output_log(wildcards.sample)
