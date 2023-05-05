import requests
import hashlib
import os
from tqdm.auto import tqdm


class Model():
    def __init__(
        self,
        name: str,
        url_weights: str,
        hash_weights: str,
        url_config: str,
        hash_config: str,
        output_folder: str,
    ) -> None:
        
        self.name = name
        self.url_weights = url_weights
        self.hash_weights = hash_weights
        self.url_config = url_config
        self.hash_config = hash_config
        self.output_folder = output_folder
        self.file_name_weights = self._generic_get_filename(self.url_weights)
        self.file_name_cfg = self._generic_get_filename(self.url_config)

        # Create output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if self.check_hash() == False:
            self.download()

    
    def get_weights(self) -> str:
        return self.output_folder + self.file_name_weights
    

    def get_config(self) -> str:
        return self.output_folder + self.file_name_cfg


    def _generic_get_filename(self, url: str) -> str:
        r = requests.get(url, allow_redirects=True, stream=True)
        cd = r.headers.get('content-disposition')
        if not cd:
            return url.split('/')[-1]
        fname = cd.split('filename=')[1].split(';')[0]
        return fname


    def _generic_download(self, url: str, output_path: str) -> None:
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        file_name = self._generic_get_filename(url)
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Donwloading {file_name}")
        with open(output_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

    
    def _generic_check_hash(self, existing_file: str, wanted_hash: str) -> bool:
        if not os.path.exists(existing_file):
            return False

        with open(existing_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != wanted_hash:
            print(f"Hash of {existing_file} is not correct. Want {wanted_hash}, got {file_hash}")
            return False
        return True


    def download(self) -> None:
        # Download weights
        self._generic_download(self.url_weights, self.get_weights())
        # Download config
        self._generic_download(self.url_config, self.get_config())


    def check_hash(self) -> bool:
        return self._generic_check_hash(self.get_weights(), self.hash_weights) and \
            self._generic_check_hash(self.get_config(), self.hash_config)
        

class YOLOv2(Model):
    def __init__(self) -> None:
        super().__init__(
            name = 'yolov2',
            url_weights = 'https://pjreddie.com/media/files/yolov2.weights',
            hash_weights = '70d89ba2e180739a1c700a9ff238e354',
            url_config = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg',
            hash_config = '781358d436479d867771b2f8ba3f7c07',
            output_folder = 'yolov2/'
        )


class YOLOv3(Model):
    def __init__(self) -> None:
        super().__init__(
            name = 'yolov3',
            url_weights = 'https://pjreddie.com/media/files/yolov3.weights',
            hash_weights = 'c84e5b99d0e52cd466ae710cadf6d84c',
            url_config = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            hash_config = 'b969a43a848bbf26901643b833cfb96c',
            output_folder = 'yolov3/'
        )


class EfficientDetD0(Model):
    def __init__(self) -> None:
        super().__init__(
            name = 'efficientdet-d0',
            url_weights = 'https://www.dropbox.com/s/9mqp99fd2tpuqn6/efficientdet-d0.pb?dl=1',
            hash_weights = '53cbf15c04fb67e2d67a42967dea9505',
            url_config = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/efficientdet-d0.pbtxt',
            hash_config = '3d6ed94ef97ab0d4dfd6471e3dbd5cc1',
            output_folder = 'efficientdet-d0/'
        )


if __name__ == '__main__':
    YOLOv2()
    YOLOv3()
    EfficientDetD0()