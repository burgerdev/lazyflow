###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#		   http://ilastik.org/license/
###############################################################################
#Python
import gc
import os
import time
import threading
import weakref
import platform

import logging
logger = logging.getLogger(__name__)

#external dependencies
import psutil

#lazyflow
from lazyflow.utility import OrderedSignal, Singleton
import lazyflow

this_process = psutil.Process(os.getpid())


def memoryUsage():
    '''
    get current memory usage in bytes
    '''
    return this_process.memory_info().rss


def memoryUsagePercentage():
    '''
    get the percentage of (memory in use) / (allowed memory use)
    
    Note: the return value is obviously non-negative, but if the user specified
    memory limit is smaller than the amount of memory actually available, this
    value can be larger than 1.
    '''
    return (memoryUsage() * 100.0) / getAvailableRamBytes()


def getAvailableRamBytes():
    '''
    get the amount of memory, in bytes, that lazyflow is allowed to use
    
    Note: When a user specified setting (e.g. via .ilastikrc) is not available,
    the function will try to estimate how much memory is available after
    subtracting known overhead. Overhead estimation is currently only available
    on Mac.
    '''
    if "Darwin" in platform.system():
        # only Mac and BSD have the wired attribute, which we can use to
        # assess available RAM more precisely
        ram = psutil.virtual_memory().total - psutil.virtual_memory().wired
    else:
        ram = psutil.virtual_memory().total
    if lazyflow.AVAILABLE_RAM_MB != 0:
        # AVAILABLE_RAM_MB is the total RAM the user wants us to limit ourselves to.
        ram = min(ram, lazyflow.AVAILABLE_RAM_MB * 1024**2)
    return ram


class MemInfoNode:
    def __init__(self):
        self.type = None
        self.id = None
        # memory used by this cache and all children, cumulated
        self.usedMemory = None
        self.dtype = None
        self.roi = None
        self.fractionOfUsedMemoryDirty = None
        self.lastAccessTime = None
        self.name = None
        # list of children's MemInfoNodes
        self.children = []


_refresh_interval = .1
_managerCondition = threading.Condition()


class CacheMemoryManager(threading.Thread):
    '''
    class for the management of cache memory

    TODO: cache cleanup documentation

    Usage:
    This manager is a singleton - just call its constructor somewhere and you
    will get a reference to the *only* running memory management thread.

    Interface:
    The manager provides a signal you can subscribe to
    >>> mgr = ArrayCacheManager
    >>> mgr.totalCacheMemory.subscribe(print)
    which emits the size of all managed caches, combined, in regular intervals.

    The update interval (for the signal and for automated cache release) can
    be set with a call to a class method
    >>> ArrayCacheManager.setRefreshInterval(5)
    the interval is measured in seconds. Each change of refresh interval
    triggers cleanup.
    '''
    __metaclass__ = Singleton

    totalCacheMemory = OrderedSignal()

    loggingName = __name__ + ".ArrayCacheMemoryMgr"
    logger = logging.getLogger(loggingName)
    traceLogger = logging.getLogger("TRACE." + loggingName)

    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

        self._caches = weakref.WeakSet()

        # maximum percentage of *allowed memory* used
        self._max_usage = 85
        # target usage percentage
        self._target_usage = 70
        self._last_usage = memoryUsagePercentage()
        self.start()

    def addCache(self, cache):
        """
        add a cache to be managed

        Caches are kept with weak references, so there is no need to remove
        them.
        """
        # late import to prevent import loop
        from lazyflow.operators.opCache import OpManagedCache
        assert isinstance(cache, OpManagedCache),\
            "Only OpManagedCache can be managed by CacheMemoryManager"
        self._caches.add(cache)

    def getChildren(self):
        '''
        implementation of OpManagedCache detail to use manager as root node
        '''
        return self._caches 

    def run(self):
        while True:
            try:
                # check current memory state
                current_usage_percentage = memoryUsagePercentage()
                if current_usage_percentage <= self._max_usage:
                    self._wait()
                    continue
                # we need a cache cleanup
                caches = list(ravel(self))
                while current_usage_percentage > self._target_usage and caches:
                    c = caches.pop(0)
                    self.logger.debug("Cleaning up cache '{}'".format(c.name))
                    c.freeMemory()
            except Exception as e:
                self.logger.error(str(e))

            # done cleaning up
            self._wait()

    def _wait(self):
        '''
        sleep for _refresh_interval seconds or until woken up
        '''
        # can't use context manager because of error messages at shutdown
        _managerCondition.acquire()
        _managerCondition.wait(_refresh_interval)
        if _managerCondition is not None:
            # no idea how that happens
            _managerCondition.release()

    @classmethod
    def setRefreshInterval(cls, t):
        with _managerCondition:
            _refresh_interval = t
            _managerCondition.notifyAll()
        
    
def ravel(cache, sort_key=lambda c: c.lastAccessTime(), postfix=True):
    '''
    iterate over the tree starting at node 'cache'
    
    The 'sort_key' callable, if present, is used to sort the list of children
    in ascending order.
    If 'postfix' is True, parents will be appended after their children,
    otherwise they will be prepended.
    The combined defaults result in the iterator being totally ordered by
    lastAccessTime().
    '''
    if not postfix:
        yield cache
    for c in sorted(cache.getChildren(), key=sort_key):
        for k in ravel(c, sort_key=sort_key, postfix=postfix):
            yield k
    if postfix:
        yield cache
