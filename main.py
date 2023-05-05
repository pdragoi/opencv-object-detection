from streamlink import Streamlink
from camera import ThreadedCamera


def main():
    
    # src = "https://digilive.rcs-rds.ro/digilivedge/nadlac_desktop.stream/chunklist_w703475052.m3u8"
    src = "https://www.youtube.com/watch?v=G05wmWFDtSo"

    session = Streamlink()

    streams = session.streams(url=src)
    print(list(streams.keys()))
    stream_link = streams["best"].to_url()

    threaded_camera = ThreadedCamera(stream_link)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass


if __name__ == '__main__':
    main()
