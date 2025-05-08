-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jun 11, 2022 at 08:33 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `1numberhelmetdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `entrytb`
--

CREATE TABLE `entrytb` (
  `id` bigint(20) NOT NULL auto_increment,
  `VehicleNo` varchar(250) NOT NULL,
  `UserName` varchar(250) NOT NULL,
  `Date` varchar(250) NOT NULL,
  `Time` varchar(250) NOT NULL,
  `FineAmount` varchar(250) NOT NULL,
  `Status` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;

--
-- Dumping data for table `entrytb`
--

INSERT INTO `entrytb` (`id`, `VehicleNo`, `UserName`, `Date`, `Time`, `FineAmount`, `Status`) VALUES
(3, '21BH2345AA', 'abi', '2022-06-11', '13:42:51', '500', 'Paid');

-- --------------------------------------------------------

--
-- Table structure for table `regtb`
--

CREATE TABLE `regtb` (
  `id` bigint(10) NOT NULL auto_increment,
  `VehicleNo` varchar(250) NOT NULL,
  `Name` varchar(250) NOT NULL,
  `Gender` varchar(250) NOT NULL,
  `Age` varchar(250) NOT NULL,
  `Email` varchar(250) NOT NULL,
  `Phone` varchar(250) NOT NULL,
  `Address` varchar(250) NOT NULL,
  `UserName` varchar(250) NOT NULL,
  `Password` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `regtb`
--

INSERT INTO `regtb` (`id`, `VehicleNo`, `Name`, `Gender`, `Age`, `Email`, `Phone`, `Address`, `UserName`, `Password`) VALUES
(1, '21BH2345AA', 'abi', 'female', '20', 'sangeeth5535@gmail.com', '9361342440', 'No 16, Samnath Plaza, Madurai Main Road, Melapudhur', 'abi', 'abi');
