from jina import Flow

if __name__ == '__main__':
    f = Flow().add(
        name='video_encoder',
        uses='Executors/VideoExecutor/config.yml',
    ).add(
        name='text_encoder',
        uses='Executors/TextExecutor/config.yml',
    ).add(
        name='indexer',
        uses='Executors/IndexerExecutor/config.yml',
    )

    with f:
        f.block()
