import os
import subprocess
from subprocess import DEVNULL, STDOUT
import time
import requests
import random
import threading
from multiprocessing import Pool

#### Fake User Agent stuff:

RVS = [
    "11.0",
    "40.0",
    "42.0",
    "43.0",
    "47.0",
    "50.0",
    "52.0",
    "53.0",
    "54.0",
    "61.0",
    "66.0",
    "67.0",
    "69.0",
    "70.0",
]

ENC = ["; U", "; N", "; I", ""]

MACS = [str(i) for i in range(6, 15)] + ["14_4", "10_1", "9_3"]


def rv():
    if random.random() < 0.1:
        return ""
    else:
        return "; rv:" + random.choice(RVS)


# OS


def linux():
    window = ["X11", "Wayland", "Unknown"]
    arch = ["i686", "x86_64", "arm"]
    distro = [
        "",
        "; Ubuntu/14.10",
        "; Ubuntu/16.10",
        "; Ubuntu/19.10",
        "; Ubuntu",
        "; Fedora",
    ]
    return (
        random.choice(window)
        + random.choice(ENC)
        + "; Linux "
        + random.choice(arch)
        + random.choice(distro)
    )


def windows():
    nt = ["5.1", "5.2", "6.0", "6.1", "6.2", "6.3", "6.4", "9.0", "10.0"]
    arch = ["; WOW64", "; Win64; x64", "; ARM", ""]
    trident = ["", "; Trident/5.0", "; Trident/6.0", "; Trident/7.0"]
    return (
        "Windows "
        + random.choice(nt)
        + random.choice(ENC)
        + random.choice(arch)
        + random.choice(trident)
    )


def mac():
    return "Macintosh; Intel Mac OS X 10_" + random.choice(MACS) + random.choice(ENC)


# Browser


def presto():
    p = [
        "2.12.388",
        "2.12.407",
        "22.9.168",
        "2.9.201",
        "2.8.131",
        "2.7.62",
        "2.6.30",
        "2.5.24",
    ]
    v = ["10.0", "11.0", "11.1", "11.5", "11.6", "12.00", "12.14", "12.16"]
    return f"Presto/{random.choice(p)} Version/{random.choice(v)}"


def product():
    opera = ["Opera/9.80", "Opera/12.0"]
    if random.random() < 0.05:
        return "Mozilla/5.0"
    else:
        return random.choice(opera)


def _os():
    r = random.random()
    if r < 0.6:
        _os = windows()
    elif r < 0.9:
        _os = linux()
    else:
        _os = mac()
    return f"({_os}{rv()})"


def browser(prod):
    if "Opera" in prod:
        return presto()
    else:
        return "like Gecko"


# Agent


def get_fake_agent():
    prod = product()
    return f"{prod} {_os()} {browser(prod)}"


### TOR stuff:


class TorNotInstalledError(Exception):
    pass


class TorRequests:
    def __init__(self, n_procs=10, start_port=9060):
        self.check_installed()
        self.n_procs = n_procs
        self.start_port = start_port
        self.ports = list(
            range(self.start_port, self.start_port + (self.n_procs * 2), 2)
        )
        self.cmds, self.pids, self.conf_files, self.bad_ports = None, None, None, []
        self.init_tor()
        self.good_ports = [i for i in self.ports if i not in self.bad_ports]
        print(f"{len(self.good_ports)} / {len(self.ports)} good ports")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def check_installed(self):
        try:
            subprocess.check_output("tor -h", shell=True).decode()
        except subprocess.CalledProcessError as e:
            if "not found" in e.output.decode():
                raise TorNotInstalledError(
                    "Tor is not installed on this system. Try running `apt-get install tor` / `yum install tor`"
                )
            else:
                print(e.output.decode())
                raise e

    def get_proxies(self):
        """
        Gets a random proxy dict from the available socks ports
        """
        port = self.good_ports[random.randint(0, len(self.good_ports) - 1)]
        return {
            "http": f"socks5://localhost:{port}",
            "https": f"socks5://localhost:{port}",
        }

    def init_tor(self, quiet=True):
        self.configure_tor(quiet=quiet)
        self.check_tor()

    def configure_tor(self, quiet=True):
        os.system("sudo chmod -R 777 /etc/tor")
        cmds = []
        pids = []
        conf_files = []
        for idx, i in enumerate(self.ports):
            conf = f"SocksPort {i}\nControlPort {i + 1}\nDataDirectory /var/lib/tor{idx}\nNumCPUs {os.cpu_count()}".strip()
            filename = f"/etc/tor/torrc.{idx}"
            with open(filename, "w") as f:
                f.write(conf)
            cmd = f"sudo tor -f {filename}"
            if quiet:
                _stdout = DEVNULL
                _stderr = STDOUT
            else:
                _stdout, _stderr = None, None
            proc = subprocess.Popen(
                [cmd],
                shell=True,
                stdin=None,
                stdout=_stdout,
                stderr=_stderr,
                close_fds=True,
            )
            cmds.append(cmd)
            pids.append(proc.pid)
            conf_files.append(filename)
        for i in range(10):
            print(f"\rWaiting 10 seconds for tor processes to start. {10-i}.", end="")
            time.sleep(1)  # give some time for tor processes to start up
        self.cmds = cmds
        self.pids = pids
        self.conf_files = conf_files

    def _check_tor(self, port, proxies, no_tor_ip, timeout=5):
        try:
            tor_ip = requests.get(
                "https://ident.me", proxies=proxies, timeout=timeout
            ).text
            assert tor_ip != no_tor_ip, "TOR does not appear to be working correctly"
            print(f"TOR Process on port {port} IP: {tor_ip}")
        except Exception as e:
            print(f"port {port} is bad:")
            print("\t", e)
            self.bad_ports.append(port)

    def check_tor(self):
        """
        Validates tor is working as expected
        """
        # no tor:
        no_tor_ip = requests.get("https://ident.me").text
        print(f"\nValidating Tor connections...\nNormal IP: {no_tor_ip}")
        threads = []
        for p in self.ports:
            # using tor via proxies:
            proxies = {
                "http": f"socks5://localhost:{p}",
                "https": f"socks5://localhost:{p}",
            }
            thr = threading.Thread(target=self._check_tor, args=(p, proxies, no_tor_ip))
            thr.start()
            threads.append(thr)

        # Wait for all of threads to finish
        for x in threads:
            x.join()

    def close(self):
        # shuts down the pids associated with the tor processes
        for cmd in self.cmds:
            cmd = cmd.replace("sudo ", "")
            cmd = f'sudo pkill -f "^{cmd}"'
            try:
                subprocess.check_call(cmd, stdout=DEVNULL, stderr=STDOUT, shell=True)
            except subprocess.CalledProcessError as e:
                pass
        for conf_file in self.conf_files:
            try:
                os.remove(conf_file)
            except Exception as e:
                print(e)

    def get(self, *args, **kwargs):
        return self._req(requests.get, *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._req(requests.post, *args, **kwargs)

    def curl_dl(self, url, out_path, quiet=True, timeout=10, max_retries=5):
        port = self.good_ports[random.randint(0, len(self.good_ports) - 1)]
        if quiet:
            prefix = "curl -Ls"
        else:
            prefix = "curl -L"
        cmd = f"{prefix} --socks5 localhost:{port} --socks5-hostname localhost:{port} '{url}' -o '{str(out_path)}' --max-time {timeout}"
        try:
            x = subprocess.check_output(cmd, shell=True)
            return 1
        except subprocess.CalledProcessError as e:
            if max_retries > 0:
                if not quiet:
                    print(
                        f"Download for {url} failed - retrying {max_retries} more times"
                    )
                return self.curl_dl(
                    url,
                    out_path,
                    quiet=quiet,
                    timeout=timeout,
                    max_retries=max_retries - 1,
                )
            else:
                print(f"Download for {url} failed. No more retries.")
                return 0

    def _req(self, fn, *args, **kwargs):
        headers = {"User-Agent": get_fake_agent()}
        if kwargs.get("headers", {}):
            headers.update(kwargs["headers"])
        proxies = self.get_proxies()
        return fn(*args, **kwargs, headers=headers, proxies=proxies)


if __name__ == "__main__":
    n_procs = 20
    url = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Indians_NW_of_South_Carolina.jpg"

    with TorRequests(n_procs) as tr:

        def do(i):
            tr.curl_dl(url, f"/home/opc/wiki_image_text/wiki_{i}.jpg")

        start = time.time()
        with Pool(n_procs) as p:
            times = p.map(do, range(50))
        print(f"Took {time.time() - start}s")
