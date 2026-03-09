/* Dataset support for valkey-benchmark
 *
 * Copyright Valkey Contributors.
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 */

#include "fmacros.h"

#include "valkey-benchmark-dataset.h"
#include "zmalloc.h"
#include <valkey/valkey.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Internal constants */
#define PLACEHOLDER_COUNT 10
#define PLACEHOLDER_LEN 12
#define XML_TAG_OVERHEAD 16 /* Space for XML tag syntax ("<", ">", "</", ">") and null terminator */

static const char *PLACEHOLDERS[PLACEHOLDER_COUNT] = {
    "__rand_int__", "__rand_1st__", "__rand_2nd__", "__rand_3rd__", "__rand_4th__",
    "__rand_5th__", "__rand_6th__", "__rand_7th__", "__rand_8th__", "__rand_9th__"};

/* Forward declarations */
static bool datasetBuildFieldMap(dataset *ds, sds *template_argv, int template_argc);
static sds getFieldValue(const char *row, int column_index, char delimiter);
static sds getXmlFieldValue(const char *xml_doc, const char *field_name);
static sds formatBytes(size_t bytes);
static bool csvDiscoverFields(dataset *ds);
static bool scanXmlFieldsFromFile(dataset *ds, const char *xml_root_element);
static bool scanXmlFields(const char *doc_start, const char *doc_end, dataset *ds, const char *start_root_tag, const char *end_root_tag);
static bool loadXmlDataset(dataset *ds, const char *xml_root_element, int verbose);
static bool loadDatasetRecords(dataset *ds, int verbose);
static bool parseNpyHeader(FILE *fp, int *rows, int *cols, bool *is_structured);
static bool parseStructuredNpy(FILE *fp, dataset *ds, int *rows);
static bool shouldStopLoading(dataset *ds);
static int findFieldIndex(dataset *ds, const char *field_name, size_t field_name_len);
static const char *extractDatasetFieldValue(dataset *ds, int field_idx, int record_index);
static sds replaceOccurrence(sds processed_arg, const char *pos, const char *replacement);
static sds processFieldsInArg(dataset *ds, sds arg, int record_index);
static sds processRandPlaceholdersForDataSet(sds cmd, _Atomic uint64_t *seq_key, int replace_placeholders, int keyspacelen, int sequential_replacement);

/* Streaming buffer context for XML document scanning */
typedef struct xmlStreamBuffer {
    char *buffer;
    size_t capacity;
    size_t used;
} xmlStreamBuffer;

static xmlStreamBuffer *createXmlStreamBuffer(void);
static void freeXmlStreamBuffer(xmlStreamBuffer *sb);
static bool growXmlStreamBuffer(xmlStreamBuffer *sb);
static size_t readIntoXmlStreamBuffer(xmlStreamBuffer *sb, FILE *fp);

dataset *datasetInit(const char *filename, const char *xml_root_element, int max_documents, int has_field_placeholders, sds *template_argv, int template_argc, int verbose) {
    if (!filename) return NULL;

    dataset *ds = zcalloc(sizeof(dataset));
    if (!ds) return NULL;

    ds->filename = filename;
    ds->xml_root_element = xml_root_element;
    ds->max_documents = max_documents;

    /* Validate XML parameters */
    if (strstr(filename, ".xml") && !xml_root_element) {
        fprintf(stderr, "Error: XML dataset requires --xml-root-element parameter\n");
        zfree(ds);
        return NULL;
    }

    /* Detect format */
    if (strstr(filename, ".npy")) {
        ds->format = DATASET_FORMAT_NPY;
        ds->delimiter = 0;
    } else if (strstr(filename, ".csv")) {
        ds->format = DATASET_FORMAT_CSV;
        ds->delimiter = ',';
    } else if (strstr(filename, ".tsv")) {
        ds->format = DATASET_FORMAT_TSV;
        ds->delimiter = '\t';
    } else if (strstr(filename, ".xml")) {
        ds->format = DATASET_FORMAT_XML;
        ds->delimiter = 0;
    } else {
        ds->format = DATASET_FORMAT_CSV;
        ds->delimiter = ',';
    }

    /* Discover fields */
    if (ds->format == DATASET_FORMAT_NPY) {
        /* Check if structured by peeking at header */
        FILE *peek_fp = fopen(filename, "r");
        if (peek_fp) {
            int rows, cols;
            bool is_structured_peek;
            if (parseNpyHeader(peek_fp, &rows, &cols, &is_structured_peek)) {
                if (is_structured_peek) {
                    /* Structured NPY - parse dtype for fields */
                    parseStructuredNpy(peek_fp, ds, &rows);
                } else {
                    /* Simple NPY - single vector field */
                    ds->field_names = zmalloc(1 * sizeof(sds));
                    ds->field_names[0] = sdsnew("vector");
                    ds->field_count = 1;
                }
            }
            fclose(peek_fp);
        }
        if (!ds->field_names) {
            /* Fallback if peek failed */
            ds->field_names = zmalloc(1 * sizeof(sds));
            ds->field_names[0] = sdsnew("vector");
            ds->field_count = 1;
        }
    } else if (ds->format == DATASET_FORMAT_XML) {
        if (!scanXmlFieldsFromFile(ds, xml_root_element)) goto error;
    } else {
        if (!csvDiscoverFields(ds)) goto error;
    }

    /* Build field map if needed (BEFORE loading) */
    if (has_field_placeholders && template_argv && template_argc > 0) {
        if (!datasetBuildFieldMap(ds, template_argv, template_argc)) goto error;
    } else {
        ds->used_field_count = ds->field_count;
    }

    /* Load data with correct field count */
    if (ds->format == DATASET_FORMAT_XML) {
        if (!loadXmlDataset(ds, xml_root_element, verbose)) goto error;
    } else {
        /* Unified loader for CSV/TSV/NPY */
        if (!loadDatasetRecords(ds, verbose)) goto error;
    }

    return ds;

error:
    datasetFree(ds);
    return NULL;
}

void datasetFree(dataset *ds) {
    if (!ds) return;

    if (ds->field_names) {
        for (int i = 0; i < ds->field_count; i++) {
            sdsfree(ds->field_names[i]);
        }
        zfree(ds->field_names);
    }

    if (ds->field_map) {
        zfree(ds->field_map);
    }
    
    if (ds->field_offsets) {
        zfree(ds->field_offsets);
    }
    
    if (ds->field_sizes) {
        zfree(ds->field_sizes);
    }

    if (ds->records) {
        for (size_t i = 0; i < ds->record_count; i++) {
            if (ds->records[i].fields) {
                for (int j = 0; j < ds->used_field_count; j++) {
                    sdsfree(ds->records[i].fields[j]);
                }
                zfree(ds->records[i].fields);
            }
        }
        zfree(ds->records);
    }

    zfree(ds);
}

bool datasetBuildFieldMap(dataset *ds, sds *template_argv, int template_argc) {
    if (!ds) return false;

    ds->field_map = zmalloc(ds->field_count * sizeof(int));
    ds->used_field_count = 0;

    for (int i = 0; i < ds->field_count; i++) {
        ds->field_map[i] = -1;
    }

    for (int arg_idx = 0; arg_idx < template_argc; arg_idx++) {
        const char *arg = template_argv[arg_idx];
        const char *field_pos = strstr(arg, FIELD_PREFIX);

        while (field_pos) {
            const char *field_start = field_pos + FIELD_PREFIX_LEN;
            const char *field_end = strstr(field_start, FIELD_SUFFIX);
            if (!field_end) break;

            size_t field_name_len = field_end - field_start;
            sds field_name = sdsnewlen(field_start, field_name_len);

            int field_idx = -1;
            for (int k = 0; k < ds->field_count; k++) {
                if (!strcmp(field_name, ds->field_names[k])) {
                    field_idx = k;
                    break;
                }
            }

            if (field_idx == -1) {
                fprintf(stderr, "Error: Field placeholder '__field:%s__' not found in dataset fields\n", field_name);
                fprintf(stderr, "Available fields: ");
                for (int j = 0; j < ds->field_count; j++) {
                    fprintf(stderr, "%s%s", ds->field_names[j], (j < ds->field_count - 1) ? ", " : "\n");
                }
                sdsfree(field_name);
                return false;
            }

            if (ds->field_map[field_idx] == -1) {
                ds->field_map[field_idx] = ds->used_field_count++;
            }

            sdsfree(field_name);
            field_pos = strstr(field_end + FIELD_SUFFIX_LEN, FIELD_PREFIX);
        }
    }

    return true;
}

size_t datasetGetRecordCount(dataset *ds) {
    return ds ? ds->record_count : 0;
}

void datasetReportMemory(dataset *ds) {
    if (!ds) return;

    size_t total_memory = 0;
    for (size_t i = 0; i < ds->record_count; i++) {
        for (int j = 0; j < ds->used_field_count; j++) {
            total_memory += sdslen(ds->records[i].fields[j]);
        }
    }
    sds size_str = formatBytes(total_memory);
    fprintf(stderr, "Dataset: %zu documents (%s)\n", ds->record_count, size_str);
    sdsfree(size_str);
}

sds datasetGenerateCommand(dataset *ds, int record_index, sds *template_argv, int template_argc, _Atomic uint64_t *seq_key, int replace_placeholders, int keyspacelen, int sequential_replacement) {
    if (!ds || !template_argv) return NULL;

    sds *processed_argv = zmalloc(template_argc * sizeof(sds));
    size_t *argvlen = zmalloc(template_argc * sizeof(size_t));
    
    for (int i = 0; i < template_argc; i++) {
        processed_argv[i] = processFieldsInArg(ds, sdsdup(template_argv[i]), record_index);
        argvlen[i] = sdslen(processed_argv[i]);  /* Binary-safe lengths */
    }

    char *cmd = NULL;
    int len = valkeyFormatCommandArgv(&cmd, template_argc, (const char **)processed_argv, argvlen);
    if (len == -1) {
        for (int i = 0; i < template_argc; i++) {
            sdsfree(processed_argv[i]);
        }
        zfree(processed_argv);
        zfree(argvlen);
        if (cmd) {
            free(cmd);
        }
        return NULL;
    }
    sds result = sdsnewlen(cmd, len);
    free(cmd);

    result = processRandPlaceholdersForDataSet(result, seq_key, replace_placeholders,
                                               keyspacelen, sequential_replacement);

    for (int i = 0; i < template_argc; i++) {
        sdsfree(processed_argv[i]);
    }
    zfree(processed_argv);
    zfree(argvlen);

    return result;
}

/* XML streaming buffer helpers */
static xmlStreamBuffer *createXmlStreamBuffer(void) {
    xmlStreamBuffer *sb = zmalloc(sizeof(xmlStreamBuffer));
    sb->capacity = 4 * 1024 * 1024;
    sb->buffer = zmalloc(sb->capacity);
    sb->used = 0;
    return sb;
}

static void freeXmlStreamBuffer(xmlStreamBuffer *sb) {
    if (!sb) return;
    zfree(sb->buffer);
    zfree(sb);
}

static bool growXmlStreamBuffer(xmlStreamBuffer *sb) {
    size_t new_capacity = sb->capacity * 2;
    char *new_buffer = zrealloc(sb->buffer, new_capacity);
    if (!new_buffer) return false;
    sb->buffer = new_buffer;
    sb->capacity = new_capacity;
    return true;
}

static size_t readIntoXmlStreamBuffer(xmlStreamBuffer *sb, FILE *fp) {
    size_t space_available = sb->capacity - sb->used;
    if (space_available == 0) {
        if (!growXmlStreamBuffer(sb)) return 0;
        space_available = sb->capacity - sb->used;
    }
    size_t bytes_read = fread(sb->buffer + sb->used, 1, space_available, fp);
    sb->used += bytes_read;
    return bytes_read;
}

static sds formatBytes(size_t bytes) {
    if (bytes < 1024) {
        return sdscatprintf(sdsempty(), "%zu bytes", bytes);
    } else if (bytes < 1024 * 1024) {
        return sdscatprintf(sdsempty(), "%.2f KB", bytes / 1024.0);
    } else if (bytes < 1024 * 1024 * 1024) {
        return sdscatprintf(sdsempty(), "%.2f MB", bytes / (1024.0 * 1024.0));
    } else {
        return sdscatprintf(sdsempty(), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}

/* Parse a single dtype field entry like ('name', 'S50') or ('name', '<f4', (256,))
 * Returns field size in bytes, or 0 on failure. */
static size_t parseDtypeFieldSize(const char *type_str, size_t type_len) {
    /* Skip leading whitespace, quotes, angle brackets, endian markers */
    while (type_len > 0 && (*type_str == '\'' || *type_str == '"' ||
                             *type_str == '<'  || *type_str == '>' ||
                             *type_str == '='  || *type_str == '|' ||
                             *type_str == ' ')) {
        type_str++;
        type_len--;
    }
    if (type_len == 0) return 0;

    char kind = type_str[0];
    size_t elem_size = 0;

    if (kind == 'S' || kind == 'U' || kind == 'V') {
        /* String/bytes/void: number after kind is byte count */
        elem_size = (size_t)atoi(type_str + 1);
        if (kind == 'U') elem_size *= 4; /* Unicode: 4 bytes per char */
    } else if (kind == 'f') {
        elem_size = (size_t)atoi(type_str + 1);
    } else if (kind == 'i' || kind == 'u') {
        elem_size = (size_t)atoi(type_str + 1);
    } else if (kind == 'b') {
        elem_size = 1;
    }

    return elem_size;
}

/* Dynamic structured NPY parser - reads field layout from the dtype descriptor in the header.
 * Works with any structured NPY file regardless of field types or count. */
static bool parseStructuredNpy(FILE *fp, dataset *ds, int *rows) {
    fseek(fp, 0, SEEK_SET);

    char magic[6];
    fread(magic, 1, 6, fp);

    unsigned char version[2];
    fread(version, 1, 2, fp);

    unsigned int header_len;
    if (version[0] == 1) {
        unsigned short len16;
        fread(&len16, 1, 2, fp);
        header_len = len16;
    } else {
        fread(&header_len, 1, 4, fp);
    }

    char *header = zmalloc(header_len + 1);
    fread(header, 1, header_len, fp);
    header[header_len] = '\0';

    /* Get row count from shape */
    char *shape_start = strstr(header, "'shape':");
    if (!shape_start) shape_start = strstr(header, "\"shape\":");
    if (!shape_start || !strchr(shape_start, '(')) {
        zfree(header);
        return false;
    }
    sscanf(strchr(shape_start, '('), "(%d,)", rows);

    /* Find the dtype list: 'descr': [(...), (...), ...] */
    char *descr_start = strstr(header, "'descr':");
    if (!descr_start) descr_start = strstr(header, "\"descr\":");
    if (!descr_start) { zfree(header); return false; }

    char *list_start = strchr(descr_start, '[');
    char *list_end = strchr(descr_start, ']');
    if (!list_start || !list_end || list_start >= list_end) {
        zfree(header);
        return false;
    }

    /* Count fields (count opening parens at top level) */
    int field_count = 0;
    for (char *p = list_start + 1; p < list_end; p++) {
        if (*p == '(') field_count++;
    }
    if (field_count == 0 || field_count > MAX_DATASET_FIELDS) {
        zfree(header);
        return false;
    }

    ds->field_names   = zmalloc(field_count * sizeof(sds));
    ds->field_offsets = zmalloc(field_count * sizeof(size_t));
    ds->field_sizes   = zmalloc(field_count * sizeof(size_t));
    ds->field_count   = 0;

    size_t current_offset = 0;
    char *p = list_start + 1;

    while (p < list_end && ds->field_count < field_count) {
        /* Find next '(' */
        while (p < list_end && *p != '(') p++;
        if (p >= list_end) break;
        p++; /* skip '(' */

        /* Extract field name (quoted string) */
        while (p < list_end && *p != '\'' && *p != '"') p++;
        if (p >= list_end) break;
        char quote = *p++;
        char *name_start = p;
        while (p < list_end && *p != quote) p++;
        size_t name_len = p - name_start;
        if (name_len == 0 || name_len > MAX_FIELD_NAME_LEN) { p++; continue; }
        p++; /* skip closing quote */

        /* Skip comma and whitespace */
        while (p < list_end && (*p == ',' || *p == ' ')) p++;

        /* Extract type string (quoted) */
        while (p < list_end && *p != '\'' && *p != '"') p++;
        if (p >= list_end) break;
        char tquote = *p++;
        char *type_start = p;
        while (p < list_end && *p != tquote) p++;
        size_t type_len = p - type_start;
        p++; /* skip closing quote */

        /* Parse element size from type string */
        size_t elem_size = parseDtypeFieldSize(type_start, type_len);
        if (elem_size == 0) { continue; }

        /* Check for shape tuple e.g. , (256,) after type */
        size_t total_size = elem_size;
        char *after_type = p;
        while (after_type < list_end && (*after_type == ',' || *after_type == ' ')) after_type++;
        if (after_type < list_end && *after_type == '(') {
            /* Shape tuple: multiply element size */
            int dim = atoi(after_type + 1);
            if (dim > 0) total_size = elem_size * (size_t)dim;
        }

        /* Store field */
        int idx = ds->field_count;
        ds->field_names[idx]   = sdsnewlen(name_start, name_len);
        ds->field_offsets[idx] = current_offset;
        ds->field_sizes[idx]   = total_size;
        current_offset += total_size;
        ds->field_count++;

        /* Advance past closing ')' */
        while (p < list_end && *p != ')') p++;
        if (p < list_end) p++;
    }

    ds->record_size = current_offset;

    zfree(header);

    if (ds->field_count == 0) {
        fprintf(stderr, "Error: No fields parsed from structured NPY dtype\n");
        return false;
    }

    return true;
}

/* NPY format parser - reads numpy array header */
static bool parseNpyHeader(FILE *fp, int *rows, int *cols, bool *is_structured) {
    char magic[6];
    if (fread(magic, 1, 6, fp) != 6) return false;
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Error: Invalid NPY magic bytes\n");
        return false;
    }
    
    unsigned char version[2];
    if (fread(version, 1, 2, fp) != 2) return false;
    
    unsigned int header_len;
    if (version[0] == 1) {
        unsigned short len16;
        if (fread(&len16, 1, 2, fp) != 2) return false;
        header_len = len16;
    } else {
        if (fread(&header_len, 1, 4, fp) != 4) return false;
    }
    
    char *header = zmalloc(header_len + 1);
    if (fread(header, 1, header_len, fp) != header_len) {
        zfree(header);
        return false;
    }
    header[header_len] = '\0';
    
    char *dtype_start = strstr(header, "'descr':");
    if (!dtype_start) dtype_start = strstr(header, "\"descr\":");
    *is_structured = (dtype_start && strchr(dtype_start, '[') != NULL);
    
    char *shape_start = strstr(header, "'shape':");
    if (!shape_start) shape_start = strstr(header, "\"shape\":");
    if (!shape_start) {
        fprintf(stderr, "Error: NPY header missing 'shape'\n");
        zfree(header);
        return false;
    }
    
    char *tuple_start = strchr(shape_start, '(');
    if (!tuple_start) {
        zfree(header);
        return false;
    }
    
    if (*is_structured) {
        if (sscanf(tuple_start, "(%d,)", rows) != 1) {
            fprintf(stderr, "Error: Could not parse structured NPY shape\n");
            zfree(header);
            return false;
        }
        *cols = 0;
    } else {
        if (sscanf(tuple_start, "(%d, %d)", rows, cols) != 2) {
            fprintf(stderr, "Error: Could not parse NPY shape\n");
            zfree(header);
            return false;
        }
    }
    
    zfree(header);
    return true;
}

static bool shouldStopLoading(dataset *ds) {
    if (ds->max_documents > 0 && (int)ds->record_count >= ds->max_documents) {
        return true;
    }
    return false;
}

static sds getFieldValue(const char *row, int column_index, char delimiter) {
    int current_col = 0;
    const char *start = row;
    const char *p = row;
    int in_quotes = 0;

    while (*p) {
        if (*p == '"') {
            in_quotes = !in_quotes;
        } else if (*p == delimiter && !in_quotes) {
            if (current_col == column_index) {
                size_t len = p - start;
                if (len > 0 && start[0] == '"' && p[-1] == '"') {
                    start++;
                    len -= 2;
                }
                return sdsnewlen(start, len);
            }
            current_col++;
            start = p + 1;
        }
        p++;
    }

    if (current_col == column_index) {
        size_t len = p - start;
        if (len > 0 && start[0] == '"' && p[-1] == '"') {
            start++;
            len -= 2;
        }
        return sdsnewlen(start, len);
    }

    return sdsempty();
}

static sds getXmlFieldValue(const char *xml_doc, const char *field_name) {
    size_t field_len = strlen(field_name);
    if (field_len > MAX_FIELD_NAME_LEN) {
        return sdsempty();
    }

    char start_tag_prefix[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD], end_tag[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD];
    snprintf(start_tag_prefix, sizeof(start_tag_prefix), "<%s", field_name);
    snprintf(end_tag, sizeof(end_tag), "</%s>", field_name);

    const char *tag_start = strstr(xml_doc, start_tag_prefix);
    if (!tag_start) return sdsempty();

    const char *tag_end = strchr(tag_start, '>');
    if (!tag_end) return sdsempty();

    if (tag_end > tag_start && tag_end[-1] == '/') {
        return sdsempty();
    }

    const char *content_start = tag_end + 1;
    const char *closing_tag = strstr(content_start, end_tag);
    if (!closing_tag) return sdsempty();

    size_t content_len = closing_tag - content_start;
    return sdsnewlen(content_start, content_len);
}

static bool csvDiscoverFields(dataset *ds) {
    FILE *fp = fopen(ds->filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open dataset file: %s\n", ds->filename);
        return false;
    }

    char *line = NULL;
    size_t len = 0;
    if (getline(&line, &len, fp) == -1) {
        fprintf(stderr, "Cannot read header from dataset file\n");
        free(line);
        fclose(fp);
        return false;
    }

    len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
    if (len > 1 && line[len - 2] == '\r') line[len - 2] = '\0';

    int count;
    char delim_str[2] = {ds->delimiter, '\0'};
    sds *temp = sdssplitlen(line, strlen(line), delim_str, 1, &count);

    ds->field_names = zmalloc(count * sizeof(sds));
    for (int i = 0; i < count; i++) {
        ds->field_names[i] = sdsdup(temp[i]);
    }
    sdsfreesplitres(temp, count);
    ds->field_count = count;

    free(line);
    fclose(fp);
    return true;
}

static bool scanXmlFields(const char *doc_start, const char *doc_end, dataset *ds, const char *start_root_tag, const char *end_root_tag) {
    char field_names[MAX_DATASET_FIELDS][MAX_FIELD_NAME_LEN + 1];
    int field_count = 0;
    int root_start_tag_len = strlen(start_root_tag);
    int root_end_tag_len = strlen(end_root_tag);

    const char *current_pos = doc_start;
    while ((current_pos = strchr(current_pos, '<')) != NULL && current_pos < doc_end) {
        if (current_pos[1] == '/' || current_pos[1] == '!' ||
            !strncmp(current_pos, start_root_tag, root_start_tag_len) ||
            !strncmp(current_pos, end_root_tag, root_end_tag_len)) {
            current_pos++;
            continue;
        }

        const char *tag_end = strchr(current_pos, '>');
        if (!tag_end || tag_end >= doc_end) break;

        const char *field_start = current_pos + 1;
        const char *field_name_end = field_start;

        while (field_name_end < tag_end && *field_name_end != ' ' && *field_name_end != '\t') {
            field_name_end++;
        }

        size_t field_name_len = field_name_end - field_start;

        if (field_name_len == 0 || field_name_len > MAX_FIELD_NAME_LEN) {
            current_pos = tag_end + 1;
            continue;
        }

        int is_duplicate = 0;
        for (int i = 0; i < field_count; i++) {
            if (strlen(field_names[i]) == field_name_len &&
                !memcmp(field_names[i], field_start, field_name_len)) {
                is_duplicate = 1;
                break;
            }
        }

        if (!is_duplicate) {
            if (field_count >= MAX_DATASET_FIELDS) {
                fprintf(stderr, "Error: Dataset contains more than %d fields (limit exceeded)\n", MAX_DATASET_FIELDS);
                return false;
            }
            memcpy(field_names[field_count], field_start, field_name_len);
            field_names[field_count][field_name_len] = '\0';
            field_count++;
        }

        current_pos = tag_end + 1;
    }

    if (field_count == 0) return false;

    ds->field_names = zmalloc(field_count * sizeof(sds));
    for (int i = 0; i < field_count; i++) {
        ds->field_names[i] = sdsnew(field_names[i]);
    }
    ds->field_count = field_count;

    return true;
}

static bool scanXmlFieldsFromFile(dataset *ds, const char *xml_root_element) {
    FILE *fp = fopen(ds->filename, "r");
    if (!fp) return false;

    char start_tag_prefix[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD], start_tag[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD], end_tag[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD];
    size_t end_tag_len;
    snprintf(start_tag_prefix, sizeof(start_tag_prefix), "<%s", xml_root_element);
    snprintf(start_tag, sizeof(start_tag), "<%s>", xml_root_element);
    snprintf(end_tag, sizeof(end_tag), "</%s>", xml_root_element);
    end_tag_len = strlen(end_tag);

    xmlStreamBuffer *sb = createXmlStreamBuffer();

    while (1) {
        size_t bytes_read = readIntoXmlStreamBuffer(sb, fp);
        if (sb->used == 0) break;

        /* Search for complete first document */
        const char *tag_prefix_pos = strstr(sb->buffer, start_tag_prefix);
        if (!tag_prefix_pos) {
            if (bytes_read == 0) break;
            continue;
        }

        const char *tag_end_pos = strchr(tag_prefix_pos, '>');
        if (!tag_end_pos || tag_end_pos >= sb->buffer + sb->used) {
            if (bytes_read == 0) break;
            continue;
        }

        const char *doc_start = tag_prefix_pos;
        const char *doc_end = NULL;

        /* Search for closing tag from current position onward */
        for (size_t i = (tag_end_pos - sb->buffer); i + end_tag_len <= sb->used; i++) {
            if (strncmp(sb->buffer + i, end_tag, end_tag_len) == 0) {
                doc_end = sb->buffer + i + end_tag_len;
                break;
            }
        }

        if (!doc_end) {
            if (bytes_read == 0) break;
            continue;
        }

        /* Found complete first document - scan fields and exit */
        bool result = scanXmlFields(doc_start, doc_end, ds, start_tag, end_tag);
        freeXmlStreamBuffer(sb);
        fclose(fp);
        return result;
    }

    freeXmlStreamBuffer(sb);
    fclose(fp);
    return false;
}

static bool loadXmlDataset(dataset *ds, const char *xml_root_element, int verbose) {
    FILE *fp = fopen(ds->filename, "r");
    if (!fp) return false;

    char start_tag_prefix[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD], start_tag[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD], end_tag[MAX_FIELD_NAME_LEN + XML_TAG_OVERHEAD];
    size_t end_tag_len;
    snprintf(start_tag_prefix, sizeof(start_tag_prefix), "<%s", xml_root_element);
    snprintf(start_tag, sizeof(start_tag), "<%s>", xml_root_element);
    snprintf(end_tag, sizeof(end_tag), "</%s>", xml_root_element);
    end_tag_len = strlen(end_tag);

    xmlStreamBuffer *sb = createXmlStreamBuffer();
    bool fields_discovered = false;
    size_t capacity = 1000;

    ds->records = zmalloc(sizeof(datasetRecord) * capacity);

    if (verbose) {
        fprintf(stderr, "Loading XML dataset from %s...\n", ds->filename);
    }

    if (ds->field_names && ds->field_count > 0) {
        fields_discovered = true;
        if (verbose) {
            fprintf(stderr, "Using %d fields: ", ds->field_count);
            for (int i = 0; i < ds->field_count; i++) {
                fprintf(stderr, "%s%s", ds->field_names[i], (i < ds->field_count - 1) ? ", " : "\n");
            }
        }
    }

    while (!shouldStopLoading(ds)) {
        size_t bytes_read = readIntoXmlStreamBuffer(sb, fp);
        if (sb->used == 0) break;

        size_t scan_pos = 0;
        while (scan_pos < sb->used) {
            const char *doc_start = NULL;
            const char *tag_open_end = NULL;

            for (size_t i = scan_pos; i < sb->used; i++) {
                if (sb->buffer[i] == '<' && i + 1 < sb->used &&
                    strncmp(sb->buffer + i, start_tag_prefix, strlen(start_tag_prefix)) == 0) {
                    doc_start = sb->buffer + i;

                    for (size_t j = i + 1; j < sb->used; j++) {
                        if (sb->buffer[j] == '>') {
                            tag_open_end = sb->buffer + j;
                            break;
                        }
                    }
                    break;
                }
            }

            if (!doc_start || !tag_open_end) break;

            const char *doc_end = NULL;
            for (size_t i = (tag_open_end - sb->buffer); i + end_tag_len <= sb->used; i++) {
                if (strncmp(sb->buffer + i, end_tag, end_tag_len) == 0) {
                    doc_end = sb->buffer + i + end_tag_len;
                    break;
                }
            }

            if (!doc_end) break;

            size_t doc_len = doc_end - doc_start;

            if (!fields_discovered) {
                if (!scanXmlFields(doc_start, doc_end, ds, start_tag, end_tag)) {
                    fprintf(stderr, "No XML fields discovered\n");
                    freeXmlStreamBuffer(sb);
                    fclose(fp);
                    return false;
                }
                fields_discovered = true;

                if (verbose) {
                    fprintf(stderr, "Discovered %d fields: ", ds->field_count);
                    for (int i = 0; i < ds->field_count; i++) {
                        fprintf(stderr, "%s%s", ds->field_names[i], (i < ds->field_count - 1) ? ", " : "\n");
                    }
                }
            }

            if (ds->record_count >= capacity) {
                capacity *= 2;
                ds->records = zrealloc(ds->records, sizeof(datasetRecord) * capacity);
            }

            datasetRecord *record = &ds->records[ds->record_count];
            record->fields = zmalloc(sizeof(sds) * ds->used_field_count);

            sds doc_str = sdsnewlen(doc_start, doc_len);
            if (ds->field_map) {
                /* With field mapping - only load used fields */
                for (int i = 0; i < ds->field_count; i++) {
                    if (ds->field_map[i] >= 0) {
                        record->fields[ds->field_map[i]] = getXmlFieldValue(doc_str, ds->field_names[i]);
                    }
                }
            } else {
                /* No field mapping - load all fields directly */
                for (int i = 0; i < ds->field_count; i++) {
                    record->fields[i] = getXmlFieldValue(doc_str, ds->field_names[i]);
                }
            }
            sdsfree(doc_str);

            ds->record_count++;

            if (verbose && ds->record_count % 1000 == 0) {
                fprintf(stderr, "\rLoaded %zu documents...", ds->record_count);
                fflush(stderr);
            }

            scan_pos = doc_end - sb->buffer;
        }

        if (scan_pos > 0 && scan_pos < sb->used) {
            size_t remaining = sb->used - scan_pos;
            memmove(sb->buffer, sb->buffer + scan_pos, remaining);
            sb->used = remaining;
        } else if (scan_pos == sb->used) {
            sb->used = 0;
        }

        if (bytes_read == 0 && (sb->used == 0 || scan_pos == 0)) {
            break;
        }
    }

    if (verbose) {
        fprintf(stderr, "\rLoaded %zu documents%*s\n", ds->record_count, 20, "");
    }

    freeXmlStreamBuffer(sb);
    fclose(fp);
    return true;
}

/* Unified record loader for CSV/TSV/NPY formats */
static bool loadDatasetRecords(dataset *ds, int verbose) {
    FILE *fp = fopen(ds->filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open dataset file: %s\n", ds->filename);
        return false;
    }

    /* Format-specific header handling */
    char *line = NULL;
    size_t line_len = 0;
    int npy_cols = 0;
    size_t vec_size = 0;
    
    bool is_structured = false;
    
    if (ds->format == DATASET_FORMAT_NPY) {
        int npy_rows;
        if (!parseNpyHeader(fp, &npy_rows, &npy_cols, &is_structured)) {
            fclose(fp);
            return false;
        }
        
        if (!is_structured) {
            /* Simple NPY */
            vec_size = npy_cols * sizeof(float);
            if (verbose) {
                fprintf(stderr, "Loading NPY: %d rows × %d dims\n", npy_rows, npy_cols);
            }
        } else if (verbose) {
            fprintf(stderr, "Loading structured NPY: %d records\n", npy_rows);
        }
    } else {
        /* CSV/TSV: Skip header line */
        if (getline(&line, &line_len, fp) == -1) {
            fprintf(stderr, "Cannot read CSV header\n");
            free(line);
            fclose(fp);
            return false;
        }
    }

    /* UNIFIED: Allocate records */
    size_t capacity = 1000;
    ds->records = zmalloc(sizeof(datasetRecord) * capacity);

    /* UNIFIED: Build load indices for CSV field mapping */
    int *load_indices = NULL;
    int load_count = 0;
    if (ds->field_map && ds->format != DATASET_FORMAT_NPY) {
        load_indices = zmalloc(ds->used_field_count * sizeof(int));
        for (int i = 0; i < ds->field_count; i++) {
            if (ds->field_map[i] >= 0) {
                load_indices[load_count++] = i;
            }
        }
    }

    /* UNIFIED: Load loop */
    while (!shouldStopLoading(ds)) {
        /* Format-specific read */
        bool read_success = false;
        
        if (ds->format == DATASET_FORMAT_NPY) {
            if (ds->record_count >= capacity) {
                capacity *= 2;
                ds->records = zrealloc(ds->records, sizeof(datasetRecord) * capacity);
            }
            
            datasetRecord *record = &ds->records[ds->record_count];
            
            if (is_structured) {
                /* Structured: Read full record, extract fields by offset */
                char *record_buf = zmalloc(ds->record_size);
                if (fread(record_buf, 1, ds->record_size, fp) != ds->record_size) {
                    zfree(record_buf);
                    break;
                }
                
                /* Extract each field, strip null padding from string fields */
                record->fields = zmalloc(sizeof(sds) * ds->field_count);
                for (int i = 0; i < ds->field_count; i++) {
                    char *field_data = record_buf + ds->field_offsets[i];
                    size_t field_size = ds->field_sizes[i];
                    
                    /* Detect string fields by size (not vector/float) and strip nulls */
                    if (field_size < 256 && field_size != 4) {
                        /* String field - find actual length by searching for first null */
                        size_t actual_len = 0;
                        while (actual_len < field_size && field_data[actual_len] != '\0') {
                            actual_len++;
                        }
                        record->fields[i] = sdsnewlen(field_data, actual_len);
                    } else {
                        /* Binary field (vector/float) - use full size */
                        record->fields[i] = sdsnewlen(field_data, field_size);
                    }
                }
                zfree(record_buf);
                read_success = true;
            } else {
                /* Simple: Single vector field */
                record->fields = zmalloc(sizeof(sds) * 1);
                record->fields[0] = sdsnewlen(NULL, vec_size);
                if (fread(record->fields[0], 1, vec_size, fp) == vec_size) {
                    read_success = true;
                }
            }
        } else {
            /* CSV/TSV text read */
            if (getline(&line, &line_len, fp) == -1) break;
            if (line[0] == '\0' || line[0] == '\n') continue;

            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
            if (len > 1 && line[len - 2] == '\r') line[len - 2] = '\0';

            if (ds->record_count >= capacity) {
                capacity *= 2;
                ds->records = zrealloc(ds->records, sizeof(datasetRecord) * capacity);
            }

            datasetRecord *record = &ds->records[ds->record_count];
            record->fields = zmalloc(sizeof(sds) * ds->used_field_count);

            if (ds->field_map) {
                for (int j = 0; j < load_count; j++) {
                    int orig_idx = load_indices[j];
                    int mapped_idx = ds->field_map[orig_idx];
                    record->fields[mapped_idx] = getFieldValue(line, orig_idx, ds->delimiter);
                }
            } else {
                for (int i = 0; i < ds->field_count; i++) {
                    record->fields[i] = getFieldValue(line, i, ds->delimiter);
                }
            }
            read_success = true;
        }

        if (!read_success) break;
        
        ds->record_count++;

        /* UNIFIED: Progress logging */
        if (verbose && ds->record_count % 10000 == 0) {
            fprintf(stderr, "\rLoaded %zu records...", ds->record_count);
            fflush(stderr);
        }
    }

    /* UNIFIED: Cleanup */
    if (verbose) {
        fprintf(stderr, "\rLoaded %zu records%*s\n", ds->record_count, 20, "");
    }
    
    if (load_indices) zfree(load_indices);
    if (line) free(line);
    fclose(fp);
    return true;
}

static int findFieldIndex(dataset *ds, const char *field_name, size_t field_name_len) {
    for (int k = 0; k < ds->field_count; k++) {
        if (strlen(ds->field_names[k]) == field_name_len &&
            !memcmp(ds->field_names[k], field_name, field_name_len)) {
            return ds->field_map ? ds->field_map[k] : k;
        }
    }
    return -1;
}

static const char *extractDatasetFieldValue(dataset *ds, int field_idx, int record_index) {
    return ds->records[record_index].fields[field_idx];
}

static sds replaceOccurrence(sds processed_arg, const char *pos, const char *replacement) {
    size_t offset = pos - processed_arg;
    size_t replacement_len = strlen(replacement);
    size_t total_len = offset + replacement_len + (sdslen(processed_arg) - offset - PLACEHOLDER_LEN);

    sds result = sdsnewlen(NULL, total_len);
    char *p = result;

    memcpy(p, processed_arg, offset);
    p += offset;

    memcpy(p, replacement, replacement_len);
    p += replacement_len;

    const char *after_start = pos + PLACEHOLDER_LEN;
    size_t after_len = sdslen(processed_arg) - offset - PLACEHOLDER_LEN;
    memcpy(p, after_start, after_len);

    sdsfree(processed_arg);
    return result;
}

static sds processFieldsInArg(dataset *ds, sds arg, int record_index) {
    if (!strstr(arg, FIELD_PREFIX)) return arg;

    /* Loop through all field placeholders in the argument */
    while (strstr(arg, FIELD_PREFIX)) {
        const char *field_pos = strstr(arg, FIELD_PREFIX);
        const char *field_start = field_pos + FIELD_PREFIX_LEN;
        const char *field_end = strstr(field_start, FIELD_SUFFIX);
        if (!field_end) break;

        size_t field_name_len = field_end - field_start;
        int field_idx = findFieldIndex(ds, field_start, field_name_len);
        if (field_idx == -1) break;

        const char *field_value = extractDatasetFieldValue(ds, field_idx, record_index);
        size_t field_value_len = sdslen(ds->records[record_index].fields[field_idx]);
        size_t before_len = field_pos - arg;
        const char *after_start = field_end + FIELD_SUFFIX_LEN;

        sds result = sdsnewlen(arg, before_len);
        result = sdscatlen(result, field_value, field_value_len);  /* Binary-safe */
        result = sdscat(result, after_start);

        sdsfree(arg);
        arg = result;
    }

    return arg;
}

static sds processRandPlaceholdersForDataSet(sds cmd, _Atomic uint64_t *seq_key, int replace_placeholders, int keyspacelen, int sequential_replacement) {
    if (!replace_placeholders || keyspacelen == 0) return cmd;

    for (int ph = 0; ph < PLACEHOLDER_COUNT; ph++) {
        if (!strstr(cmd, PLACEHOLDERS[ph])) continue;

        uint64_t shared_key = 0;
        int generate_shared_key = (ph != 0);

        if (generate_shared_key) {
            if (sequential_replacement) {
                shared_key = atomic_fetch_add_explicit(&seq_key[ph], 1, memory_order_relaxed);
            } else {
                shared_key = (uint64_t)random();
            }
            shared_key %= keyspacelen;
        }

        size_t search_offset = 0;
        char *pos;
        while ((pos = strstr(cmd + search_offset, PLACEHOLDERS[ph])) != NULL) {
            uint64_t key = generate_shared_key ? shared_key : 0;

            if (!generate_shared_key) {
                if (sequential_replacement) {
                    key = atomic_fetch_add_explicit(&seq_key[ph], 1, memory_order_relaxed);
                } else {
                    key = (uint64_t)random();
                }
                key %= keyspacelen;
            }

            char key_str[24];
            snprintf(key_str, sizeof(key_str), "%012llu", (unsigned long long)key);

            size_t offset = pos - cmd;
            cmd = replaceOccurrence(cmd, pos, key_str);
            search_offset = offset + PLACEHOLDER_LEN;
        }
    }

    return cmd;
}
