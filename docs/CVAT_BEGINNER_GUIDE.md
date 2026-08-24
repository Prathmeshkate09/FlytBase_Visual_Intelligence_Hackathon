# CVAT beginner guide for FlytBase Level 1

This guide creates corrected ground truth for the 89-frame development clip. The
model-generated boxes are only a starting point. They become ground truth only
after every frame and every relevant road user has been reviewed.

## What to upload

Upload only these two files:

1. Video while creating the task:

   ```text
   C:\Users\PRATHAMESH\Documents\Codex\2026-08-22\kn\work\cvat_upload\level1_v4_dev_3s\source_video_89f.mp4
   ```

2. Seed annotations after the task exists:

   ```text
   C:\Users\PRATHAMESH\Documents\Codex\2026-08-22\kn\work\cvat_upload\level1_v4_dev_3s\annotations.xml
   ```

The label schema to paste into CVAT is `config/cvat_labels_level1.json`.

Do **not** upload either complete video from `D:\Flybase`. They are the source
archive for extracting future development, validation, and test clips. Do not
upload `level1_v4_heldout_3s` yet; keeping it unseen prevents tuning on the test
clip. Use `annotations.xml`, not the convenience ZIP, for the initial import.

## Part 1 - Create the task

1. Open `https://app.cvat.ai/`, create an account or sign in, and open **Tasks**.
2. Select **+** and then **Create new task**.
3. Name it `FlytBase-L1-development-89f-GT`.
4. Leave **Project** empty.
5. In **Labels**, open the **Raw** tab.
6. Open `config/cvat_labels_level1.json`, copy the complete JSON array, paste it
   into the Raw editor, and select **Done**.

   The text attribute deliberately uses `"values": [""]`. CVAT Online's Raw
   editor rejects an empty `values` array even though `v4_track_id` is free text.
   Every label also declares `"type": "any"`; without an explicit type, CVAT
   Online rejects the label as `unknown label type "undefined"`.
7. Select **My computer** and upload `source_video_89f.mp4` from the path above.
8. Open **Advanced configuration** and use:

   | Setting | Value |
   |---|---:|
   | Start frame | `0` |
   | Stop frame | `88` |
   | Frame step | `1` |
   | Segment size | `89` |
   | Overlap | `0` |
   | Image quality | `100` |
   | Chunk size | `4` |
   | Use ZIP chunks | Off, if the option is shown |

9. Select **Submit and open**. Wait until the task finishes processing.

The clip already contains exactly 89 frames, but explicit start/stop settings
make the split auditable. Quality 100 helps preserve tiny pedestrians and
bicycles, and chunk size 4 is appropriate for 4K annotation playback.

## Part 2 - Import the model seed before editing

1. On the task details page, open **Actions** and select **Upload annotations**.
   The same command may be available in the annotation job's **Menu**.
2. Select format **CVAT for video 1.1**.
3. Upload `annotations.xml` from the development folder.
4. Wait for the import to complete, then open the single job.
5. Save once before beginning the review.

Import the seed before doing any manual work because uploading annotations can
replace annotations already in the task.

## Part 3 - What counts as a road user

Include:

- cars, motorcycles, bicycles, buses, goods vehicles, and pedestrians;
- objects moving on the road, footpath, crossing, or visible approach;
- temporarily stopped vehicles;
- parked/stationary vehicles on the road or curb that are part of the traffic
  scene; and
- an object from its first usable visibility until it exits or becomes fully
  unresolvable.

Exclude:

- roofs, signs, road markings, trees, shadows, construction material, and empty
  space;
- pictures of vehicles on advertisements; and
- objects wholly outside the defined traffic scene.

Use `ignore` only when an actual road user is too truncated, too occluded, or too
small to label reliably. Do not use motion as a validity filter.

## Part 4 - Class rules

Use these rules consistently:

- `car`: passenger car, taxi, hatchback, sedan, or SUV.
- `lgv`: clearly visible van, pickup, or light delivery vehicle.
- `hgv`: unmistakably heavy or articulated/multi-axle goods vehicle.
- `truck`: goods vehicle visible as a truck when LGV versus HGV cannot be
  determined reliably from the pixels.
- `bus`: bus, minibus, or coach when visually distinguishable.
- `motorcycle`: motorcycle or motor scooter.
- `bicycle`: pedal bicycle and its rider as one tracked road user.
- `pedestrian`: a person on foot.
- `ignore`: real but genuinely unresolvable road user; never a false positive.

When the pixels do not support a fine-grained goods-vehicle class, do not guess.
Consistent uncertainty is more useful than a confident wrong label.

## Part 5 - Review every frame

Work in four passes and save frequently.

### Pass A - Remove false tracks and add missing objects

1. Start at frame 0 and inspect the entire frame at high zoom.
2. Delete tracks on roofs, buildings, signs, trees, road marks, or empty space.
3. Add every missing road user with **Rectangle -> Track**, not a one-frame
   shape.
4. Repeat through frame 88. Do not review only the seeded boxes.

### Pass B - Repair identity continuity

For each physical object, follow it from first appearance to exit:

- It must have one CVAT track, even if the detector class changed.
- If two seed tracks describe the same physical object before and after a gap,
  use **Merge** (`M`) after confirming they are the same object.
- If one track jumps from one physical object to another, use **Split** and
  repair the affected tracks.
- If an object is partly hidden, mark it **Occluded** and adjust the box.
- If it is fully invisible, mark the track **Outside** (`O`) on the first hidden
  frame and return the same track on the first visible frame. If two tracks were
  created, merge them only after visual confirmation.
- Mark **Outside** on the first frame after the object really exits.

Never invent a box for a fully invisible object. Identity can remain continuous
without pretending that an unobserved box is a real detection.

### Pass C - Correct boxes and classes

- Keep each rectangle tight around the visible road user.
- Add keyframes when position, scale, direction, or visibility changes. CVAT
  interpolates between keyframes, so check the intermediate frames.
- A stationary object still needs keyframes if drone motion shifts its image
  position.
- Correct the class at the track level using the rules above.
- Treat `v4_track_id` and `seed_state` as audit metadata from the model. The
  final CVAT track identity is the ground-truth identity.

### Pass D - Final frame-by-frame audit

For frames 0 through 88, verify:

- every relevant physical road user has one box or is explicitly outside;
- every visible box belongs to a real object;
- no physical object has two simultaneous IDs;
- no track jumps to a different physical object;
- stopped and parked road users are included;
- pedestrians and bicycles on footpaths are included; and
- all entry, exit, occlusion, crossing, and turning events are correct.

Set the job status to completed only after this pass. The auxiliary
`review_status` attribute is useful for notes, but it does not replace the full
frame audit.

## Part 6 - Export the corrected truth

Download and keep three exports from **Actions -> Export task dataset** (or the
job **Menu -> Export as a dataset**):

1. **CVAT for video 1.1** - master editable annotation.
2. **MOT 1.1** - tracking identities for IDF1/HOTA evaluation.
3. **COCO 1.0** - per-frame boxes/classes for detection precision and recall.

Use names such as:

```text
FlytBase_L1_dev89_corrected_CVAT.zip
FlytBase_L1_dev89_corrected_MOT.zip
FlytBase_L1_dev89_corrected_COCO.zip
```

Place all three downloads in:

```text
C:\Users\PRATHAMESH\Documents\Codex\2026-08-22\kn\work\cvat_exports\level1_v4_dev_3s_corrected
```

Do not overwrite the original seed package. Once the three files are present,
the next engineering step is to validate their structure, run the Level-1
detector/tracker evaluation, and separate remaining failures by pipeline layer.

## Recovery rules

- If the seed import fails, do not annotate manually in a broken task. Capture
  the exact CVAT error and keep the task ID.
- If labels do not match, compare the Raw label schema with
  `config/cvat_labels_level1.json`.
- If CVAT reports `attribute values must be a non-empty array`, make sure every
  `v4_track_id` definition contains `"values": [""]`, not `"values": []`.
- If CVAT reports `unknown label type "undefined"`, make sure every label
  includes `"type": "any"`.
- If the video has any frame count other than 89, stop and recreate the task.
- If browser playback is slow, reduce the visible canvas zoom or close other
  heavy tabs; do not lower task image quality after annotation begins.

## Official CVAT references

- Task creation and Raw labels: `https://docs.cvat.ai/docs/manual/basics/create-annotation-task/`
- Video track/interpolation mode: `https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/`
- Annotation import/export: `https://docs.cvat.ai/docs/manual/advanced/import-datasets/`
- MOT tracking format: `https://docs.cvat.ai/docs/dataset_management/formats/format-mot/`
- COCO detection format: `https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/`
