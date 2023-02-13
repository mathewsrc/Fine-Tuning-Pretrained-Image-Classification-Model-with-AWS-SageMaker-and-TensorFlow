#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: No directory path provided."
    echo "Usage: $0 [-d] [-r] <directory_path>"
    exit 1
fi

delete_flag=false
recursive_flag=false

while getopts "dr" opt; do
    case $opt in
        d)
            delete_flag=true
            ;;
        r)
            recursive_flag=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

dir_path=$1

if [ ! -d "$dir_path" ]; then
    echo "Error: Provided directory path is not valid."
    exit 1
fi

echo "Converting all JPG images in $dir_path to JPEG format..."

if [ "$recursive_flag" = true ]; then
    find "$dir_path" -type f -name "*.jpg" | while read file; do
        convert "$file" "${file%.jpg}.jpeg"
        if [ "$delete_flag" = true ]; then
            rm "$file"
        fi
    done
else
    for file in "$dir_path"/*.jpg; do
        convert "$file" "${file%.jpg}.jpeg"
        if [ "$delete_flag" = true ]; then
            rm "$file"
        fi
    done
fi

echo "Done."
