# Downloads and prepares the videos and accompanying data 
# from the What's Cookin release for alignment
# Usage: dl_and_prep_data.sh [path to initial list of videos]
# Requirements: youtube-dl, ffmpeg

if [[ $# -eq 0 ]]; then
  echo "Usage: dl_and_prep_data.sh [path to csv of videos]"
  exit 0
fi

LISTPATH=$1
UNIQID=$(cat $LISTPATH | cut -d , -f 1 | sort | uniq)

WORKING_DIR="./data"
PATH_TO_INGREDIENT_SENTENCES="./all-recipecom-ingredient-sentences.txt"
PATH_TO_INSTRUCTION_SENTENCES="./all-recipecom-instruction-sentences.txt"
PATH_TO_BACKGROUND_SENTENCES="./eng-eu_web_2014_1M-sentences_cleaned.txt"
SEGMENT_LENGTH=8 # in seconds

mkdir $WORKING_DIR

echo "Downloading videos..."
mkdir $WORKING_DIR/videos
mkdir $WORKING_DIR/frames
for id in $UNIQID; do
  # save a temporary full version of the video
  youtube-dl -f mp4 -o $WORKING_DIR/videos/$id.full.mp4 https://youtube.com/watch?v=$id;

  # cut out the relevant segments and extract the frames
  for entry in $(cat $LISTPATH | grep $id ); do
    s=$(echo $entry | cut -d , -f 6); # grab seconds value
    ms=$(echo $entry | cut -d , -f 2); # grab milliseconds value
    echo $s | xargs -I {} \
    ffmpeg \
    -ss {} \
    -i $WORKING_DIR/videos/$id.full.mp4 \
    -t $SEGMENT_LENGTH -c copy $WORKING_DIR/videos/$id.$ms.mp4;
    mkdir $WORKING_DIR/frames/$id.$ms;
    ffmpeg \
    -i $WORKING_DIR/videos/$id.$ms.mp4 \
    -vf "select=not(mod(n\,10)), scale=-1:256" \
    -vsync vfr -q:v 2 -y $WORKING_DIR/frames/$id.$ms/frame_%04d.jpg;
  done

  # remove the temp full video
  rm $WORKING_DIR/videos/$id.full.mp4;
done

echo "Downloading video descriptions..."
mkdir $WORKING_DIR/descriptions
mkdir $WORKING_DIR/classify
for id in $UNIQID; do
youtube-dl \
--skip-download \
--write-description \
-o "$WORKING_DIR/descriptions/%(id)s.%(ext)s" \
https://youtube.com/watch?v=$id;
python extract_from_url.py $WORKING_DIR/descriptions/$id.description;
echo "...classifying recipe sentences..."
python sentence_classifier.py \
$PATH_TO_INGREDIENT_SENTENCES \
$PATH_TO_INSTRUCTION_SENTENCES \
$PATH_TO_BACKGROUND_SENTENCES \
$WORKING_DIR/descriptions \
$WORKING_DIR/classify;
echo "...aligning segments to sentences..."
python align_action_object_to_sentence.py \
$LISTPATH \
$WORKING_DIR/classify \
$WORKING_DIR/aligned_sentences.txt;
done

echo "Downloading speech transcripts..."
mkdir $WORKING_DIR/transcripts
for id in $UNIQID; do
  youtube-dl \
  --skip-download \
  --write-auto-sub \
  --sub-format ttml \
  -o $WORKING_DIR/transcripts/$id.ttml \
  https://youtube.com/watch?v=$id;
done
